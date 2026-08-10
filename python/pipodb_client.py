import urllib.request
import json
import time

class PipoDBClient:
    def __init__(self, host="http://localhost:8080"):
        self.host = host.rstrip('/')

    def _post(self, endpoint, data=None):
        url = f"{self.host}{endpoint}"
        payload = json.dumps(data if data else {}).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=payload, 
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def _get(self, endpoint):
        url = f"{self.host}{endpoint}"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def health(self):
        """Returns server health status."""
        return self._get("/health")

    def list_collections(self):
        """Returns list of active collections in PipoDB."""
        return self._get("/api/v1/collections")

    def create_collection(self, name, dimension=128, metric="L2", index_type="HNSW"):
        """Creates a new collection."""
        return self._post("/api/v1/collections/create", {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index_type": index_type
        })

    def drop_collection(self, name):
        """Drops an existing collection."""
        return self._post("/api/v1/collections/drop", {"name": name})

    def insert_vector(self, collection_name, vector, payload="", explicit_id=0):
        """Inserts a vector into a collection."""
        return self._post("/api/v1/vectors/insert", {
            "collection": collection_name,
            "id": explicit_id,
            "vector": vector,
            "payload": payload
        })

    def upsert_if_close(self, collection_name, vector, payload="", distance_threshold=0.05):
        """Semantic near-duplicate upsert."""
        return self._post("/api/v1/vectors/upsert", {
            "collection": collection_name,
            "vector": vector,
            "payload": payload,
            "distance_threshold": distance_threshold
        })

    def delete_vector(self, collection_name, vector_id):
        """Deletes a vector by ID."""
        return self._post("/api/v1/vectors/delete", {
            "collection": collection_name,
            "id": vector_id
        })

    def search(self, collection_name, query_vector, top_k=10, filter_key=None, filter_value=None):
        """Performs vector similarity search."""
        req = {
            "collection": collection_name,
            "vector": query_vector,
            "top_k": top_k
        }
        if filter_key and filter_value:
            req["filter_key"] = filter_key
            req["filter_value"] = filter_value
        return self._post("/api/v1/vectors/search", req)


if __name__ == "__main__":
    print("==================================================")
    print("      PipoDB Python Client SDK Demo & Test        ")
    print("==================================================")
    
    client = PipoDBClient("http://localhost:8080")
    
    try:
        health = client.health()
        print(f"\n1. Server Health: {health}")
        
        print("\n2. Creating Collection 'py_docs'...")
        c_res = client.create_collection("py_docs", dimension=3, metric="COSINE", index_type="HNSW")
        print(f"   Response: {c_res}")
        
        print("\n3. Inserting Vectors...")
        i1 = client.insert_vector("py_docs", [1.0, 1.0, 1.0], payload="Python Docs 1", explicit_id=101)
        i2 = client.insert_vector("py_docs", [2.0, 2.0, 2.0], payload="Python Docs 2", explicit_id=102)
        print(f"   Vector 101: {i1}")
        print(f"   Vector 102: {i2}")
        
        print("\n4. Performing Vector Similarity Search...")
        s_res = client.search("py_docs", [2.1, 2.1, 2.1], top_k=2)
        print(f"   Search Results: {json.dumps(s_res, indent=2)}")
        
        print("\n==================================================")
        print(" [SUCCESS] Python SDK Demo Complete!")
        print("==================================================")

    except Exception as ex:
        print(f"[ERROR] Could not connect to PipoDB server: {ex}")
