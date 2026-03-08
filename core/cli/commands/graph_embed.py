"""CLI commands for RGCN structural graph embeddings.

Commands:
    hades graph-embed train     — train RGCN on a database snapshot
    hades graph-embed embed     — get structural embedding for a node
    hades graph-embed neighbors — find k-nearest structural neighbors
"""

from __future__ import annotations

from core.cli.output import ErrorCode, error_response, success_response


def graph_train(
    database: str,
    device: str,
    epochs: int,
    patience: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    export: bool,
    export_to: str | None,
    start_time: float,
):
    """Train RGCN on a database graph."""
    try:
        from core.graph.train import RGCNTrainer, TrainConfig

        config = TrainConfig(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            lr=lr,
            epochs=epochs,
            patience=patience,
        )

        trainer = RGCNTrainer(database=database, device=device, config=config)
        trainer.load_data()
        metrics = trainer.train()

        result = {
            "database": database,
            "device": device,
            "num_nodes": trainer.graph_data["num_nodes"],
            "num_edges": trainer.graph_data["num_edges"],
            "num_relations": trainer.graph_data["num_relations"],
            "best_epoch": metrics["best_epoch"],
            "best_val_loss": round(metrics["best_val_loss"], 4),
            "test_acc": round(metrics["test_acc"], 3),
            "test_auc": round(metrics["test_auc"], 3),
            "training_seconds": round(metrics["elapsed_seconds"], 1),
            "model_path": metrics["model_path"],
        }

        if export:
            target = export_to or database
            n_exported = trainer.export_embeddings(target_database=target)
            result["exported_to"] = target
            result["nodes_exported"] = n_exported

        return success_response(
            command="graph-embed.train",
            data=result,
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="graph-embed.train",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def graph_embed_node(
    node_id: str,
    database: str,
    start_time: float,
):
    """Get structural embedding for a specific node."""
    import base64
    import json
    import os
    import urllib.request

    try:
        pw = os.environ["ARANGO_PASSWORD"]
        auth = base64.b64encode(f"root:{pw}".encode()).decode()
        base_url = f"http://127.0.0.1:8529/_db/{database}"

        # Fetch the node's structural embedding
        col, key = node_id.split("/", 1) if "/" in node_id else (None, node_id)

        if col is None:
            return error_response(
                command="graph-embed.embed",
                code=ErrorCode.QUERY_FAILED,
                message="Node ID must be in format 'collection/key'",
                start_time=start_time,
            )

        query = f"FOR d IN {col} FILTER d._key == @key RETURN {{ _id: d._id, structural_embedding: d.structural_embedding, title: d.title, name: d.name }}"
        data = json.dumps({"query": query, "bindVars": {"key": key}}).encode()
        req = urllib.request.Request(
            f"{base_url}/_api/cursor",
            data=data,
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req).read())

        if resp.get("error") or not resp.get("result"):
            return error_response(
                command="graph-embed.embed",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Node not found: {node_id}",
                start_time=start_time,
            )

        node = resp["result"][0]
        if not node.get("structural_embedding"):
            return error_response(
                command="graph-embed.embed",
                code=ErrorCode.QUERY_FAILED,
                message=f"No structural embedding for {node_id}. Run 'hades graph-embed train' first.",
                start_time=start_time,
            )

        return success_response(
            command="graph-embed.embed",
            data={
                "node_id": node_id,
                "label": node.get("title") or node.get("name") or key,
                "embedding_dim": len(node["structural_embedding"]),
                "embedding": node["structural_embedding"],
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="graph-embed.embed",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def graph_update(
    database: str,
    device: str,
    export_to: str | None,
    start_time: float,
):
    """Incremental update: reload graph, re-embed with trained model, export."""
    try:
        from core.graph.train import RGCNTrainer

        trainer = RGCNTrainer(database=database, device=device)
        cfg = trainer.load_model()
        trainer.load_data()

        target = export_to or database
        result = trainer.update_embeddings(target_database=target)
        result["database"] = database
        result["device"] = device
        result["model_config"] = cfg

        return success_response(
            command="graph-embed.update",
            data=result,
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="graph-embed.update",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def graph_neighbors(
    node_id: str,
    database: str,
    k: int,
    collections: list[str] | None,
    start_time: float,
):
    """Find k-nearest structural neighbors for a node."""
    import base64
    import json
    import os
    import urllib.request

    try:
        pw = os.environ["ARANGO_PASSWORD"]
        auth = base64.b64encode(f"root:{pw}".encode()).decode()
        base_url = f"http://127.0.0.1:8529/_db/{database}"

        col, key = node_id.split("/", 1) if "/" in node_id else (None, node_id)
        if col is None:
            return error_response(
                command="graph-embed.neighbors",
                code=ErrorCode.QUERY_FAILED,
                message="Node ID must be in format 'collection/key'",
                start_time=start_time,
            )

        # Get target embedding
        query = f"FOR d IN {col} FILTER d._key == @key RETURN d.structural_embedding"
        data = json.dumps({"query": query, "bindVars": {"key": key}}).encode()
        req = urllib.request.Request(
            f"{base_url}/_api/cursor",
            data=data,
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req).read())

        if not resp.get("result") or resp["result"][0] is None:
            return error_response(
                command="graph-embed.neighbors",
                code=ErrorCode.QUERY_FAILED,
                message=f"No structural embedding for {node_id}",
                start_time=start_time,
            )

        target_emb = resp["result"][0]
        embed_dim = len(target_emb)

        # Determine which collections to search
        if collections:
            search_cols = collections
        else:
            # Get all document collections that have structural embeddings
            q = "FOR c IN COLLECTIONS() FILTER c.type == 2 AND !STARTS_WITH(c.name, '_') " "RETURN c.name"
            data = json.dumps({"query": q}).encode()
            req = urllib.request.Request(
                f"{base_url}/_api/cursor",
                data=data,
                headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            search_cols = resp.get("result", [])

        # Search each collection for nearest neighbors
        all_neighbors = []
        for search_col in search_cols:
            query = f"""
            LET te = @target_emb
            FOR d IN {search_col}
              FILTER d.structural_embedding != null
              FILTER LENGTH(d.structural_embedding) == @dim
              FILTER d._id != @target_id
              LET sim = SUM(FOR i IN 0..@dim_minus_1 RETURN te[i] * d.structural_embedding[i])
              SORT sim DESC
              LIMIT @k
              RETURN {{
                id: d._id,
                label: d.title || d.name || d._key,
                collection: "{search_col}",
                similarity: ROUND(sim * 10000) / 10000
              }}
            """
            data = json.dumps(
                {
                    "query": query,
                    "bindVars": {
                        "target_emb": target_emb,
                        "dim": embed_dim,
                        "dim_minus_1": embed_dim - 1,
                        "target_id": node_id,
                        "k": k,
                    },
                }
            ).encode()
            req = urllib.request.Request(
                f"{base_url}/_api/cursor",
                data=data,
                headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            if resp.get("result"):
                all_neighbors.extend(resp["result"])

        # Sort globally and take top k
        all_neighbors.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = all_neighbors[:k]

        return success_response(
            command="graph-embed.neighbors",
            data={
                "query_node": node_id,
                "k": k,
                "neighbors": top_k,
            },
            start_time=start_time,
            count=len(top_k),
        )

    except Exception as e:
        return error_response(
            command="graph-embed.neighbors",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
