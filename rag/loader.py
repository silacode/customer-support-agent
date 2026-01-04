from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos
            else:
                # Look for sentence break
                period_pos = text.rfind(". ", start, end)
                if period_pos > start + chunk_size // 2:
                    end = period_pos + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def load_policies(policies_dir: str = "policies") -> list[dict]:
    """
    Load all policy documents from the policies directory.
    
    Returns:
        List of dicts with 'content', 'source', and 'title' keys
    """
    policies_path = Path(policies_dir)
    documents = []
    
    if not policies_path.exists():
        return documents
    
    for file_path in policies_path.glob("*.md"):
        content = file_path.read_text()
        
        # Extract title from first heading
        lines = content.split("\n")
        title = file_path.stem.replace("_", " ").title()
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        # Chunk the content
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "source": str(file_path),
                "title": title,
                "chunk_index": i,
            })
    
    return documents

