import tiktoken


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(
                "Chunk size must be positive."
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "Overlap must be between zero "
                "and chunk size minus one."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(
            "cl100k_base"
        )
        
    def split(self, text: str) -> list[str]:
        text = text.strip()
        
        if not text:
            return []

        tokens = self.encoding.encode(text)
        
        chunks: list[str] = []

        step = (
            self.chunk_size - self.chunk_overlap
        )
        
        for start in range(
            0,
            len(tokens),
            step,
        ):
            end = min(
                start + self.chunk_size,
                len(tokens),
            )
            
            chunk_tokens = tokens[start:end]

            chunk = self.encoding.decode(
                chunk_tokens
            ).strip()
            
            if chunk:
                chunks.append(chunk)

            if end == len(tokens):
                break
            
        return chunks