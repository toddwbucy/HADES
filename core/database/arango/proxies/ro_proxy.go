package proxies

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"time"
)

// NOTE: Defense-in-depth: The upstream ArangoDB connection should use a read-only
// user with limited permissions. This proxy provides an additional layer of protection
// but should not be the only security boundary.

func RunReadOnlyProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultROListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadOnly)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, roSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-only proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

// validateQueryViaExplain calls ArangoDB's explain endpoint to analyze the query
// and detect write operations in the execution plan, avoiding false positives from
// keywords appearing inside string literals.
func validateQueryViaExplain(upstreamSocket string, query string, bindVars map[string]interface{}) error {
	// Build explain request payload
	explainPayload := map[string]interface{}{
		"query": query,
	}
	if bindVars != nil {
		explainPayload["bindVars"] = bindVars
	}

	payloadBytes, err := json.Marshal(explainPayload)
	if err != nil {
		return fmt.Errorf("failed to marshal explain request: %w", err)
	}

	// Create HTTP client for upstream connection
	transport := newUnixTransport(upstreamSocket)
	client := &http.Client{
		Transport: transport,
		Timeout:   10 * time.Second,
	}

	// Call the explain endpoint
	req, err := http.NewRequest(http.MethodPost, "http://arangodb/_api/explain", bytes.NewReader(payloadBytes))
	if err != nil {
		return fmt.Errorf("failed to create explain request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("explain request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read explain response: %w", err)
	}

	// Parse explain response
	var explainResp struct {
		Error        bool   `json:"error"`
		ErrorMessage string `json:"errorMessage"`
		Plan         struct {
			Nodes []struct {
				Type string `json:"type"`
			} `json:"nodes"`
		} `json:"plan"`
	}

	if err := json.Unmarshal(body, &explainResp); err != nil {
		// If we can't parse the response, fall back to rejecting for safety
		return fmt.Errorf("failed to parse explain response: %w", err)
	}

	if explainResp.Error {
		// Query has syntax error or other issue - let it through to get proper error
		// from the actual query execution
		return nil
	}

	// Check for write operation nodes in the execution plan
	writeOperations := map[string]bool{
		"InsertNode":        true,
		"UpdateNode":        true,
		"UpsertNode":        true,
		"RemoveNode":        true,
		"ReplaceNode":       true,
		"ModificationNode":  true,
	}

	for _, node := range explainResp.Plan.Nodes {
		if writeOperations[node.Type] {
			return fmt.Errorf("write operation %q detected in query plan", node.Type)
		}
	}

	return nil
}

// Global variable to store upstream socket for explain calls
var roUpstreamSocket string

func allowReadOnly(r *http.Request, peek BodyPeeker) error {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return nil
	case http.MethodPost:
		if isCursorPath(r.URL.Path) {
			body, err := peek(128 * 1024)
			if err != nil {
				return err
			}
			var payload struct {
				Query    string                 `json:"query"`
				BindVars map[string]interface{} `json:"bindVars"`
			}
			if err := json.Unmarshal(body, &payload); err == nil && payload.Query != "" {
				// Use ArangoDB's explain endpoint to validate the query
				upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)
				if err := validateQueryViaExplain(upstreamSocket, payload.Query, payload.BindVars); err != nil {
					return err
				}
				return nil
			}
			// If we couldn't parse the query, reject for safety
			return fmt.Errorf("could not parse query from request body")
		}
	case http.MethodPut, http.MethodDelete:
		if isCursorPath(r.URL.Path) {
			return nil
		}
	}
	return fmt.Errorf("method %s not permitted on %s", r.Method, r.URL.Path)
}
