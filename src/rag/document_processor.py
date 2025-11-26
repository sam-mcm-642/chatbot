import json
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Process documents into chunks for embedding
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_text_file(self, file_path: str) -> str:
        """Load a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_json_file(self, file_path: str) -> List[Dict]:
        """Load JSON file (e.g., Day One export)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_markdown_file(self, file_path: str) -> str:
        """Load markdown file (e.g., Craft export)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of dicts with 'text', 'metadata', and 'chunk_id'
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip very small chunks
                continue
            
            chunk = {
                'text': chunk_text,
                'metadata': metadata or {},
                'chunk_id': len(chunks)
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_directory(self, directory: str, file_extension: str = '.txt') -> List[Dict]:
        """
        Process all files in a directory
        
        Args:
            directory: Path to directory
            file_extension: File type to process (.txt, .md, .json)
        
        Returns:
            List of processed chunks
        """
        all_chunks = []
        directory_path = Path(directory)
        
        for file_path in directory_path.glob(f"**/*{file_extension}"):
            print(f"Processing: {file_path}")
            
            # Load based on file type
            if file_extension == '.json':
                data = self.load_json_file(str(file_path))
                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        text = item.get('text', '') or item.get('content', '')
                        metadata = {k: v for k, v in item.items() if k not in ['text', 'content']}
                        metadata['source'] = str(file_path)
                        chunks = self.chunk_text(self.clean_text(text), metadata)
                        all_chunks.extend(chunks)
                elif isinstance(data, dict):
                    text = data.get('text', '') or data.get('content', '')
                    metadata = {'source': str(file_path)}
                    chunks = self.chunk_text(self.clean_text(text), metadata)
                    all_chunks.extend(chunks)
            else:
                # Text or markdown files
                text = self.load_text_file(str(file_path))
                metadata = {'source': str(file_path), 'filename': file_path.name}
                chunks = self.chunk_text(self.clean_text(text), metadata)
                all_chunks.extend(chunks)
        
        print(f"Processed {len(all_chunks)} total chunks from {directory}")
        return all_chunks
    
    def process_day_one_export(self, export_path: str) -> List[Dict]:
        """
        Process Day One journal export
        Expected format: JSON with entries
        """
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        all_chunks = []
        entries = data.get('entries', [])
        
        for entry in entries:
            text = entry.get('text', '')
            metadata = {
                'date': entry.get('creationDate', ''),
                'location': entry.get('location', {}).get('placeName', ''),
                'tags': entry.get('tags', []),
                'source': 'day_one'
            }
            
            chunks = self.chunk_text(self.clean_text(text), metadata)
            all_chunks.extend(chunks)
        
        print(f"Processed {len(all_chunks)} chunks from Day One export")
        return all_chunks
    
    def process_craft_weekly_targets(self, file_path: str) -> List[Dict]:
        """
        Process Craft weekly targets - handles both old and new formats
        
        Old format (2024): Simple list with Result section
        New format (2025): Full Challenge/Result/Reflection sections
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract date from title - handles both ::DD/MM/YY:: and other variations
        date_match = re.search(r'#\s*::(\d{2}/\d{2}/\d{2,4}):\s*Weekly Targets::', content)
        week_date = None
        if date_match:
            date_str = date_match.group(1)
            try:
                # Try DD/MM/YY format first
                if len(date_str.split('/')[-1]) == 2:
                    dt = datetime.strptime(date_str, '%d/%m/%y')
                else:
                    dt = datetime.strptime(date_str, '%d/%m/%Y')
                week_date = dt.strftime('%Y-%m-%d')
            except:
                week_date = date_str  # Keep original if parsing fails
        
        # Extract tasks and metadata
        tasks_data = self.extract_weekly_tasks(content)
        
        # Build metadata - only non-None values
        metadata = {
            'source': 'craft',
            'doc_type': 'weekly_targets',
            'filename': Path(file_path).name,
        }
        
        if week_date:
            metadata['week_date'] = week_date
        
        if tasks_data['total_tasks'] > 0:
            metadata['total_tasks'] = tasks_data['total_tasks']
            metadata['completed_tasks'] = tasks_data['completed_tasks']
            metadata['completion_rate'] = tasks_data['completion_rate']
        
        if tasks_data['high_priority_count'] > 0:
            metadata['high_priority_tasks'] = tasks_data['high_priority_count']
        
        if tasks_data['has_reflection']:
            metadata['has_reflection'] = True
        
        # Determine format (old vs new)
        has_challenge_section = '### ::Challenge::' in content
        metadata['format'] = 'new' if has_challenge_section else 'old'
        
        chunks = []
        
        # Main tasks section (always present)
        tasks_section = self.extract_tasks_section(content)
        if tasks_section:
            chunks.append({
                'text': self.clean_craft_syntax(tasks_section),
                'metadata': {**metadata, 'section': 'main_tasks'},
                'chunk_id': 0
            })
        
        # Reflection sections (only if they have content)
        reflection_sections = self.extract_reflection_sections(content)
        chunk_id = 1
        
        for section_name, section_content in reflection_sections.items():
            # Only add if there's actual content (not just empty or whitespace)
            cleaned = section_content.strip()
            if cleaned and len(cleaned) > 10:  # At least 10 chars of content
                chunks.append({
                    'text': cleaned,
                    'metadata': {**metadata, 'section': section_name},
                    'chunk_id': chunk_id
                })
                chunk_id += 1
        
        # If no structured chunks, create one from full document
        if not chunks:
            chunks = [{
                'text': self.clean_craft_syntax(content),
                'metadata': metadata,
                'chunk_id': 0
            }]
        
        return chunks
    
    def extract_tasks_section(self, content: str) -> str:
        """
        Extract just the tasks section (everything up to Low Priority or first ###)
        """
        # Find the start (after the title)
        title_end = re.search(r'#\s*::.*?::\s*\n', content)
        if not title_end:
            return content
        
        start = title_end.end()
        
        # Find the end (Low Priority marker or first ### section)
        low_priority_match = re.search(r'\*\*::Low Priority::', content[start:])
        section_match = re.search(r'###\s*::', content[start:])
        result_match = re.search(r'::Result:::', content[start:])
        
        # Use whichever comes first
        end_positions = [
            low_priority_match.start() if low_priority_match else None,
            section_match.start() if section_match else None,
            result_match.start() if result_match else None
        ]
        end_positions = [pos for pos in end_positions if pos is not None]
        
        if end_positions:
            end = start + min(end_positions)
            return content[start:end]
        else:
            return content[start:]
    
    def extract_reflection_sections(self, content: str) -> Dict[str, str]:
        """
        Extract Challenge, Result, and Reflection sections
        Only returns sections with actual content
        """
        sections = {}
        
        # Challenge section
        challenge_match = re.search(
            r'### ::Challenge::\s*\n(.*?)(?=### ::Result::|### ::Reflection::|$)',
            content,
            re.DOTALL
        )
        if challenge_match:
            challenge_text = challenge_match.group(1).strip()
            # Remove separator lines
            challenge_text = re.sub(r'^-+$', '', challenge_text, flags=re.MULTILINE).strip()
            if challenge_text:
                sections['challenge'] = challenge_text
        
        # Result section (handles both ::Result::: and ### ::Result::)
        result_match = re.search(
            r'(?:::Result::::|### ::Result::)\s*\n(.*?)(?=### ::Reflection::|### ::Challenge::|$)',
            content,
            re.DOTALL
        )
        if result_match:
            result_text = result_match.group(1).strip()
            # Remove separator lines and template questions if empty
            result_text = re.sub(r'^-+$', '', result_text, flags=re.MULTILINE).strip()
            # Remove empty template questions
            result_text = re.sub(r'^What did I do that I shouldn\'t.*?\n*', '', result_text, flags=re.MULTILINE)
            result_text = re.sub(r'^What went well\?\s*\n*', '', result_text, flags=re.MULTILINE)
            result_text = result_text.strip()
            
            if result_text:
                sections['result'] = result_text
        
        # Reflection section
        reflection_match = re.search(
            r'### ::Reflection::\s*\n(.+)',
            content,
            re.DOTALL
        )
        if reflection_match:
            reflection_text = reflection_match.group(1).strip()
            if reflection_text:
                sections['reflection'] = reflection_text
        
        return sections
    
    def extract_weekly_tasks(self, content: str) -> Dict:
        """
        Extract task statistics from weekly targets
        Handles both old and new formats
        """
        # Find all tasks (checkbox items)
        task_pattern = r'^[-*]\s*\[([ xX])\]\s*(.+)$'
        tasks = []
        
        for match in re.finditer(task_pattern, content, re.MULTILINE):
            completed = match.group(1).lower() == 'x'
            task_text = match.group(2).strip()
            
            # Check if highlighted/high priority
            is_highlighted = '::Highlight::' in task_text or '::highlight::' in task_text
            
            # Clean the task text
            clean_text = self.clean_craft_syntax(task_text)
            
            tasks.append({
                'text': clean_text,
                'completed': completed,
                'highlighted': is_highlighted
            })
        
        total = len(tasks)
        completed = sum(1 for t in tasks if t['completed'])
        highlighted = sum(1 for t in tasks if t['highlighted'])
        
        # Check if there's actual reflection content (not just template)
        has_reflection = False
        reflection_match = re.search(r'### ::Reflection::\s*\n(.+)', content, re.DOTALL)
        if reflection_match:
            reflection_text = reflection_match.group(1).strip()
            # Must have more than just whitespace
            has_reflection = len(reflection_text) > 0
        
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'completion_rate': round((completed / total * 100) if total > 0 else 0, 1),
            'high_priority_count': highlighted,
            'has_reflection': has_reflection,
            'tasks': tasks
        }
    
    def clean_craft_syntax(self, text: str) -> str:
        """Remove Craft-specific syntax like ::text::"""
        # Remove highlight markers but keep the text
        text = re.sub(r'::([Hh]ighlight):::\s*', '', text)
        text = re.sub(r'::([^:]+)::', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Remove horizontal rules
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def process_all_weekly_targets(self, directory: str) -> List[Dict]:
        """
        Process all weekly target documents in a directory
        """
        all_chunks = []
        directory_path = Path(directory)
        
        # Find all markdown files (including nested)
        files = list(directory_path.glob("**/*.md"))
        
        print(f"Found {len(files)} markdown files")
        
        for file_path in sorted(files):
            print(f"Processing: {file_path.name}")
            
            try:
                chunks = self.process_craft_weekly_targets(str(file_path))
                
                if chunks:
                    # Show info about this week
                    meta = chunks[0]['metadata']
                    week_date = meta.get('week_date', 'unknown')
                    completed = meta.get('completed_tasks', 0)
                    total = meta.get('total_tasks', 0)
                    rate = meta.get('completion_rate', 0)
                    
                    print(f"  Week: {week_date}")
                    print(f"  Tasks: {completed}/{total} ({rate}%)")
                    print(f"  Chunks created: {len(chunks)}")
                    
                    all_chunks.extend(chunks)
                else:
                    print(f"  ⚠️  No chunks created")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\nProcessed {len(all_chunks)} total chunks from weekly targets")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Process a directory of text files
    chunks = processor.process_directory("data/raw/documents", file_extension=".txt")
    
    # Save processed chunks
    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")