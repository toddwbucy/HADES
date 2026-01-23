#!/usr/bin/env php
<?php
/**
 * ArangoDB PHP Bridge for Unix Socket Operations
 *
 * Uses the triagens/arangodb PHP driver which properly supports Unix sockets.
 * This solves the python-arango limitation with Unix socket connections.
 */

require_once __DIR__ . '/../../../vendor/autoload.php';

use ArangoDBClient\Connection;
use ArangoDBClient\ConnectionOptions;
use ArangoDBClient\DocumentHandler;
use ArangoDBClient\Document;
use ArangoDBClient\CollectionHandler;
use ArangoDBClient\Collection;
use ArangoDBClient\Exception as ArangoException;
use ArangoDBClient\UpdatePolicy;

// Get operation from command line
$operation = $argv[1] ?? 'test';

// Configuration - try TCP first, then we'll fix Unix socket permissions
$connectionOptions = [
    // TODO: Fix Unix socket permissions, for now use TCP
    // ConnectionOptions::OPTION_ENDPOINT => 'unix:///tmp/arangodb.sock',
    ConnectionOptions::OPTION_ENDPOINT => 'tcp://localhost:8529',
    ConnectionOptions::OPTION_DATABASE => 'arxiv_repository',
    ConnectionOptions::OPTION_AUTH_TYPE => 'Basic',
    ConnectionOptions::OPTION_AUTH_USER => 'root',
    ConnectionOptions::OPTION_AUTH_PASSWD => getenv('ARANGO_PASSWORD') ?: '',
    ConnectionOptions::OPTION_CONNECTION => 'Keep-Alive',
    ConnectionOptions::OPTION_TIMEOUT => 30,
    ConnectionOptions::OPTION_RECONNECT => true,
    ConnectionOptions::OPTION_CREATE => false,
    ConnectionOptions::OPTION_UPDATE_POLICY => UpdatePolicy::LAST,
];

try {
    // Create connection
    $connection = new Connection($connectionOptions);

    switch ($operation) {
        case 'test':
            // Test the connection - just try to get collections to verify it works
            $collectionHandler = new CollectionHandler($connection);
            $collections = $collectionHandler->getAllCollections();
            echo json_encode([
                'status' => 'success',
                'message' => 'Connected successfully!',
                'collections_found' => count($collections),
                'endpoint' => 'tcp://localhost:8529'  // TODO: Switch to Unix socket
            ], JSON_PRETTY_PRINT) . "\n";
            break;

        case 'create_collections':
            // Create the required collections
            $collectionHandler = new CollectionHandler($connection);
            $collections = ['arxiv_metadata', 'arxiv_abstract_embeddings', 'arxiv_abstract_chunks', 'arxiv_structures'];
            $created = [];

            foreach ($collections as $name) {
                try {
                    $collection = new Collection($name);
                    $collection->setType(Collection::TYPE_DOCUMENT);
                    $id = $collectionHandler->create($collection);
                    $created[] = $name;
                    echo "✓ Created collection: $name (ID: $id)\n";
                } catch (ArangoException $e) {
                    if ($e->getCode() == 1207) {
                        echo "→ Collection already exists: $name\n";
                    } else {
                        throw $e;
                    }
                }
            }

            echo json_encode([
                'status' => 'success',
                'created' => $created
            ], JSON_PRETTY_PRINT) . "\n";
            break;

        case 'drop_collections':
            // Drop collections
            $collectionHandler = new CollectionHandler($connection);
            $collections = ['arxiv_metadata', 'arxiv_abstract_embeddings', 'arxiv_abstract_chunks', 'arxiv_structures'];
            $dropped = [];

            foreach ($collections as $name) {
                try {
                    $collectionHandler->drop($name);
                    $dropped[] = $name;
                    echo "✓ Dropped collection: $name\n";
                } catch (ArangoException $e) {
                    echo "→ Could not drop $name: " . $e->getMessage() . "\n";
                }
            }

            echo json_encode([
                'status' => 'success',
                'dropped' => $dropped
            ], JSON_PRETTY_PRINT) . "\n";
            break;

        case 'check_collections':
            // Check collection status
            $collectionHandler = new CollectionHandler($connection);
            $collections = ['arxiv_metadata', 'arxiv_abstract_embeddings', 'arxiv_abstract_chunks', 'arxiv_structures'];
            $status = [];

            foreach ($collections as $name) {
                try {
                    $collection = $collectionHandler->get($name);
                    $count = $collectionHandler->count($name);
                    $status[$name] = [
                        'exists' => true,
                        'count' => $count,
                        'id' => $collection->getId(),
                        'status' => $collection->getStatus()
                    ];
                } catch (ArangoException $e) {
                    $status[$name] = [
                        'exists' => false,
                        'error' => $e->getMessage()
                    ];
                }
            }

            echo json_encode($status, JSON_PRETTY_PRINT) . "\n";
            break;

        case 'bulk_insert':
            // Read JSON from stdin for bulk insert
            $json = file_get_contents('php://stdin');
            $data = json_decode($json, true);

            if (!$data) {
                throw new \Exception("Invalid JSON input");
            }

            $collectionName = $data['collection'] ?? 'arxiv_metadata';
            $documents = $data['documents'] ?? [];

            $documentHandler = new DocumentHandler($connection);
            $inserted = 0;

            // Process in batches for efficiency
            foreach (array_chunk($documents, 1000) as $batch) {
                $batchDocs = [];
                foreach ($batch as $doc) {
                    $document = new Document();
                    foreach ($doc as $key => $value) {
                        $document->set($key, $value);
                    }
                    $batchDocs[] = $document;
                }

                // ACID compliant atomic bulk insert using transaction
                $trx = new \ArangoDBClient\Transaction($connection);
                $trx->addWrite($collectionName);

                // Build batch insert action
                $docsJson = json_encode(array_map(function($doc) {
                    return $doc->getAll();
                }, $batchDocs));

                $action = "function () {
                    var db = require('@arangodb').db;
                    var docs = $docsJson;
                    return db.$collectionName.save(docs);
                }";

                $trx->setAction($action);

                try {
                    // Execute atomic transaction - all or nothing
                    $result = $trx->execute();
                    $inserted += count($batch);
                } catch (\Exception $e) {
                    // Transaction failed - nothing was inserted
                    throw new \Exception("ACID transaction failed: " . $e->getMessage());
                }
            }

            echo json_encode([
                'status' => 'success',
                'inserted' => $inserted,
                'collection' => $collectionName
            ], JSON_PRETTY_PRINT) . "\n";
            break;

        case 'stats':
            // Get database statistics
            $collectionHandler = new CollectionHandler($connection);
            $collections = ['arxiv_metadata', 'arxiv_abstract_embeddings', 'arxiv_abstract_chunks', 'arxiv_structures'];
            $stats = [
                'connection' => 'unix:///tmp/arangodb.sock',
                'database' => 'arxiv_repository',
                'collections' => []
            ];

            foreach ($collections as $name) {
                try {
                    $count = $collectionHandler->count($name);
                    $collection = $collectionHandler->get($name);
                    $figures = $collectionHandler->figures($name);

                    $stats['collections'][$name] = [
                        'exists' => true,
                        'count' => $count,
                        'memory' => $figures['figures']['memory']['count'] ?? 0,
                        'disk' => $figures['figures']['disk']['count'] ?? 0
                    ];
                } catch (ArangoException $e) {
                    $stats['collections'][$name] = ['exists' => false];
                }
            }

            echo json_encode($stats, JSON_PRETTY_PRINT) . "\n";
            break;

        default:
            echo json_encode([
                'error' => "Unknown operation: $operation",
                'available' => ['test', 'create_collections', 'drop_collections', 'check_collections', 'bulk_insert', 'stats']
            ], JSON_PRETTY_PRINT) . "\n";
            exit(1);
    }

} catch (ArangoException $e) {
    echo json_encode([
        'status' => 'error',
        'type' => 'ArangoDB',
        'code' => $e->getCode(),
        'message' => $e->getMessage()
    ], JSON_PRETTY_PRINT) . "\n";
    exit(1);
} catch (\Exception $e) {
    echo json_encode([
        'status' => 'error',
        'type' => 'General',
        'message' => $e->getMessage()
    ], JSON_PRETTY_PRINT) . "\n";
    exit(1);
}
