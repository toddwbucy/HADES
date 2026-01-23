package proxies

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"regexp"
)

// API endpoint patterns for strict matching
// Format: ^(/_db/[^/]+)?/<endpoint>(/.*)?$
var (
	apiImportPattern     = regexp.MustCompile(`^(/_db/[^/]+)?/_api/import(/.*)?$`)
	apiDocumentPattern   = regexp.MustCompile(`^(/_db/[^/]+)?/_api/document(/.*)?$`)
	apiCollectionPattern = regexp.MustCompile(`^(/_db/[^/]+)?/_api/collection(/.*)?$`)
	apiIndexPattern      = regexp.MustCompile(`^(/_db/[^/]+)?/_api/index(/.*)?$`)
)

// matchAPIEndpoint checks if the path matches a specific API endpoint pattern
// using anchored regex to prevent prefix/suffix/embedded path confusion
func matchAPIEndpoint(path string, pattern *regexp.Regexp) bool {
	return pattern.MatchString(path)
}

func RunReadWriteProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultRWListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadWrite)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, rwSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-write proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

func allowReadWrite(r *http.Request, peek BodyPeeker) error {
	if err := allowReadOnly(r, peek); err == nil {
		return nil
	}

	path := r.URL.Path
	switch r.Method {
	case http.MethodPost:
		if isCursorPath(path) ||
			matchAPIEndpoint(path, apiImportPattern) ||
			matchAPIEndpoint(path, apiDocumentPattern) ||
			matchAPIEndpoint(path, apiCollectionPattern) ||
			matchAPIEndpoint(path, apiIndexPattern) {
			return nil
		}
	case http.MethodPut, http.MethodPatch, http.MethodDelete:
		if matchAPIEndpoint(path, apiDocumentPattern) ||
			matchAPIEndpoint(path, apiCollectionPattern) ||
			matchAPIEndpoint(path, apiIndexPattern) {
			return nil
		}
	}

	return fmt.Errorf("method %s not permitted on %s", r.Method, path)
}
