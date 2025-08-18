#!/usr/bin/env python3
"""
Text File Splitter

This script splits a large text file into smaller files with less than 8000 words each.
It preserves paragraph boundaries and provides options for different splitting strategies.

Usage:
    python text_splitter.py input_file.txt [max_words] [output_prefix]
    
Example:
    python text_splitter.py large_document.txt 8000 split_
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple
from tkinter import Tk, filedialog


class TextSplitter:
    def __init__(self, max_chars: int = 8000, output_prefix: str = "split_"):
        """
        Initialize the text splitter.
        
        Args:
            max_chars: Maximum number of characters per output file
            output_prefix: Prefix for output filenames
        """
        self.max_chars = max_chars
        self.output_prefix = output_prefix
    
    def count_chars(self, text: str) -> int:
        """Count characters in a text string."""
        return len(text)
    
    def count_words(self, text: str) -> int:
        """Count words in a text string (for display purposes)."""
        # Split on whitespace and filter out empty strings
        words = [word for word in re.split(r'\s+', text.strip()) if word]
        return len(words)
    
    def split_by_paragraphs(self, content: str) -> List[str]:
        """
        Split content into paragraphs, preserving paragraph boundaries.
        
        Args:
            content: The text content to split
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', content.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Group paragraphs into chunks with less than max_chars.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = []
        current_char_count = 0
        
        for paragraph in paragraphs:
            paragraph_char_count = self.count_chars(paragraph)
            
            # If single paragraph exceeds max_chars, split it further
            if paragraph_char_count > self.max_chars:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_char_count = 0
                
                # Split the large paragraph by sentences
                sentences = self.split_large_paragraph(paragraph)
                chunks.extend(sentences)
                continue
            
            # Calculate the character count if we add this paragraph
            # Include the \n\n separator if this isn't the first paragraph
            separator_chars = 2 if current_chunk else 0
            total_chars = current_char_count + separator_chars + paragraph_char_count
            
            # Check if adding this paragraph would exceed the limit
            if total_chars > self.max_chars and current_chunk:
                # Save current chunk and start a new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_char_count = paragraph_char_count
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_char_count = total_chars
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def split_large_paragraph(self, paragraph: str) -> List[str]:
        """
        Split a large paragraph into smaller chunks by sentences.
        
        Args:
            paragraph: The paragraph to split
            
        Returns:
            List of text chunks
        """
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = []
        current_char_count = 0
        
        for sentence in sentences:
            sentence_char_count = self.count_chars(sentence)
            
            # If single sentence exceeds max_chars, put it in its own chunk
            # We won't split sentences in the middle to preserve readability
            if sentence_char_count > self.max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_char_count = 0
                
                # Put the entire long sentence in its own chunk
                # This preserves sentence integrity even if it exceeds the limit
                chunks.append(sentence)
                continue
            
            # Calculate character count with space separator
            separator_chars = 1 if current_chunk else 0
            total_chars = current_char_count + separator_chars + sentence_char_count
            
            if total_chars > self.max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_char_count = sentence_char_count
            else:
                current_chunk.append(sentence)
                current_char_count = total_chars
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def split_file(self, input_file: str, output_dir: str = None) -> Tuple[int, List[str]]:
        """
        Split a text file into smaller files.
        
        Args:
            input_file: Path to the input text file
            output_dir: Directory for output files (default: same as input file)
            
        Returns:
            Tuple of (number_of_files_created, list_of_output_filenames)
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read the input file
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(input_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        print(f"Input file: {input_file}")
        print(f"Total characters in input: {self.count_chars(content):,}")
        print(f"Total words in input: {self.count_words(content):,}")
        print(f"Max characters per output file: {self.max_chars:,}")
        
        # Split content into paragraphs and then into chunks
        paragraphs = self.split_by_paragraphs(content)
        chunks = self.create_chunks(paragraphs)
        
        print(f"Created {len(chunks)} chunks")
        
        # Write chunks to separate files
        output_files = []
        base_name = input_path.stem
        extension = input_path.suffix
        
        for i, chunk in enumerate(chunks, 1):
            output_filename = f"{self.output_prefix}{base_name}_part{i:03d}{extension}"
            output_path = output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            char_count = self.count_chars(chunk)
            word_count = self.count_words(chunk)
            print(f"  Part {i}: {output_filename} ({char_count:,} chars, {word_count:,} words)")
            output_files.append(str(output_path))
        
        return len(chunks), output_files


def main():
    """Main function to handle command line arguments or file dialog."""
    # Use file dialog to select input file
    root = Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(
        title="Select text file to split",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not input_file:
        print("No file selected. Exiting.")
        sys.exit(1)
    
    # Get optional parameters from command line
    max_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "split_"
    
    try:
        splitter = TextSplitter(max_chars=max_chars, output_prefix=output_prefix)
        num_files, output_files = splitter.split_file(input_file)
        
        print(f"\n✅ Successfully split into {num_files} files:")
        for file_path in output_files:
            print(f"   {file_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()