import os
import json
import kuzu
from .resolver import MedicalResolver

class KuzuLoader:
    def __init__(self, db_path: str = "data/db"):
        self.db_path = db_path
        # Kuzu expects a directory where it will create its own files. 
        # If it doesn't exist, kuzu.Database(db_path) will create it.
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self.resolver = MedicalResolver()
        self._init_schema()

    def _init_schema(self):
        """Initializes the Kùzu schema if tables don't exist."""
        try:
            self.conn.execute("CREATE NODE TABLE Entity(cui STRING, name STRING, PRIMARY KEY (cui))")
            self.conn.execute("CREATE NODE TABLE Mention(id STRING, text STRING, role STRING, PRIMARY KEY (id))")
            self.conn.execute("CREATE NODE TABLE Chunk(id STRING, source_file STRING, page_number INT, text_content STRING, PRIMARY KEY (id))")
            self.conn.execute("CREATE REL TABLE REFERS_TO(FROM Mention TO Entity)")
            self.conn.execute("CREATE REL TABLE APPEARS_IN(FROM Mention TO Chunk)")
            print("Kùzu schema initialized.")
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"Schema init warning: {e}")

    def load_chunks(self, chunks_file: str):
        """Loads extracted JSON chunks, resolves entities, and inserts into Kùzu."""
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        print(f"Loading {len(chunks)} chunks into Kùzu...")
        
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk['source_file']}_pg{chunk['page_number']}"
                
                # 1. Insert Chunk
                self.conn.execute(
                    "MERGE (c:Chunk {id: $id, source_file: $src, page_number: $pg, text_content: $txt})",
                    {"id": chunk_id, "src": chunk["source_file"], "pg": chunk["page_number"], "txt": chunk["text_content"]}
                )
                
                # 2. Process Mentions
                mentions = chunk.get("mentions", [])
                for m in mentions:
                    text = m["text"]
                    role = m["role"]
                    mention_id = f"{text}_{role}"
                    
                    # Resolve mention to canonical entity
                    resolved = self.resolver.resolve(text)
                    
                    # 3. Insert Entity (The canonical Concept) - Skip if UNKNOWN
                    if resolved["cui"] != "UNKNOWN":
                        self.conn.execute(
                            "MERGE (e:Entity {cui: $cui, name: $name})",
                            {"cui": resolved["cui"], "name": resolved["canonical_name"]}
                        )
                    
                    # 4. Insert Mention (The physical occurrence)
                    self.conn.execute(
                        "MERGE (m:Mention {id: $id, text: $txt, role: $role})",
                        {"id": mention_id, "txt": text, "role": role}
                    )
                    
                    # 5. Create Edges
                    if resolved["cui"] != "UNKNOWN":
                        self.conn.execute(
                            "MATCH (m:Mention), (e:Entity) WHERE m.id = $mid AND e.cui = $cui "
                            "MERGE (m)-[:REFERS_TO]->(e)",
                            {"mid": mention_id, "cui": resolved["cui"]}
                        )
                    
                    self.conn.execute(
                        "MATCH (m:Mention), (c:Chunk) WHERE m.id = $mid AND c.id = $cid "
                        "MERGE (m)-[:APPEARS_IN]->(c)",
                        {"mid": mention_id, "cid": chunk_id}
                    )
            self.conn.execute("COMMIT")
            print("Data load complete.")
        except Exception as e:
            self.conn.execute("ROLLBACK")
            print(f"Error during batch load, transaction rolled back: {e}")
            raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical KG Database Loader")
    parser.add_argument("--chunks", type=str, default="data/interim/raw_chunks.json", help="Path to JSON chunks file")
    parser.add_argument("--db", type=str, default="data/db", help="Path to Kùzu DB")
    
    args = parser.parse_args()
    
    try:
        loader = KuzuLoader(db_path=args.db)
        loader.load_chunks(args.chunks)
    except Exception as e:
        print(f"Database load failed: {e}")
