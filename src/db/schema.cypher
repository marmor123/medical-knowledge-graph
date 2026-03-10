// 1. Create Node Tables
CREATE NODE TABLE Entity(cui STRING, name STRING, PRIMARY KEY (cui));
CREATE NODE TABLE Mention(text STRING, role STRING, PRIMARY KEY (text, role));
CREATE NODE TABLE Chunk(id STRING, source_file STRING, page_number INT, text_content STRING, PRIMARY KEY (id));

// 2. Create Rel Tables (Edges)
CREATE REL TABLE REFERS_TO(FROM Mention TO Entity);
CREATE REL TABLE APPEARS_IN(FROM Mention TO Chunk);

// 3. (Optional) Reified Protocol Nodes for later logic
CREATE NODE TABLE ClinicalProtocol(id STRING, PRIMARY KEY (id));
CREATE REL TABLE PROTOCOL_DIAGNOSIS(FROM ClinicalProtocol TO Entity);
CREATE REL TABLE PROTOCOL_SYMPTOM(FROM ClinicalProtocol TO Entity);
