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
        Process Craft weekly targets with specific format:
        # ::DD/MM/YY: Weekly Targets::
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract date from title
        date_match = re.search(r'#\s*::(\d{2}/\d{2}/\d{2}):\s*Weekly Targets::', content)
        week_date = None
        if date_match:
            # Convert DD/MM/YY to YYYY-MM-DD for consistency
            date_str = date_match.group(1)
            try:
                dt = datetime.strptime(date_str, '%d/%m/%y')
                week_date = dt.strftime('%Y-%m-%d')
            except:
                week_date = date_str
        
        # Clean Craft-specific syntax
        cleaned_content = self.clean_craft_syntax(content)
        
        # Extract tasks and metadata
        tasks_data = self.extract_weekly_tasks(content)
        
        # Build comprehensive metadata
        metadata = {
            'source': 'craft',
            'doc_type': 'weekly_targets',
            'filename': Path(file_path).name,
            'filepath': str(file_path),
            'week_date': week_date,
            'total_tasks': tasks_data['total_tasks'],
            'completed_tasks': tasks_data['completed_tasks'],
            'completion_rate': tasks_data['completion_rate'],
            'high_priority_tasks': tasks_data['high_priority_count'],
            'has_reflection': tasks_data['has_reflection']
        }
        
        # Create chunks
        # Chunk 1: Tasks list
        tasks_section = self.extract_section(content, start='# ::', end='**::Low Priority::')
        if tasks_section:
            chunks = [{
                'text': self.clean_craft_syntax(tasks_section),
                'metadata': {**metadata, 'section': 'main_tasks'}
            }]
        else:
            chunks = []
        
        # Chunk 2: Low priority tasks if present
        low_priority = self.extract_section(content, start='**::Low Priority::', end='### ::Challenge::')
        if low_priority:
            chunks.append({
                'text': self.clean_craft_syntax(low_priority),
                'metadata': {**metadata, 'section': 'low_priority'}
            })
        
        # Chunk 3: Reflection sections
        reflection_sections = self.extract_reflection_sections(content)
        for section_name, section_content in reflection_sections.items():
            if section_content.strip():
                chunks.append({
                    'text': self.clean_craft_syntax(section_content),
                    'metadata': {**metadata, 'section': section_name}
                })
        
        # If no structured chunks, fall back to full document
        if not chunks:
            chunks = [{
                'text': cleaned_content,
                'metadata': metadata
            }]
        
        return chunks
    
    def clean_craft_syntax(self, text: str) -> str:
        """Remove Craft-specific syntax like ::text::"""
        # Remove highlight markers
        text = re.sub(r'::([^:]+)::', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def extract_weekly_tasks(self, content: str) -> Dict:
        """
        Extract task statistics from weekly targets
        """
        # Find all tasks (including low priority)
        task_pattern = r'^[-*]\s*\[([ x])\]\s*(.+)$'
        tasks = []
        
        for match in re.finditer(task_pattern, content, re.MULTILINE):
            completed = match.group(1).lower() == 'x'
            task_text = match.group(2).strip()
            
            # Check if it's highlighted/high priority
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
        
        # Check if reflection is filled out
        has_reflection = bool(re.search(r'### ::Reflection::\s*\n+\w+', content))
        
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'completion_rate': round((completed / total * 100) if total > 0 else 0, 1),
            'high_priority_count': highlighted,
            'has_reflection': has_reflection,
            'tasks': tasks
        }
    
    def extract_section(self, content: str, start: str, end: str = None) -> str:
        """Extract content between two markers"""
        start_idx = content.find(start)
        if start_idx == -1:
            return ""
        
        if end:
            end_idx = content.find(end, start_idx)
            if end_idx == -1:
                return content[start_idx:]
            return content[start_idx:end_idx]
        else:
            return content[start_idx:]
    
    def extract_reflection_sections(self, content: str) -> Dict[str, str]:
        """
        Extract Challenge, Result, and Reflection sections
        """
        sections = {}
        
        # Challenge section
        challenge = self.extract_section(content, '### ::Challenge::', '### ::Result::')
        if challenge:
            sections['challenge'] = challenge.replace('### ::Challenge::', '').strip()
        
        # Result section
        result = self.extract_section(content, '### ::Result::', '### ::Reflection::')
        if result:
            sections['result'] = result.replace('### ::Result::', '').strip()
        
        # Reflection section
        reflection_match = re.search(r'### ::Reflection::\s*(.+)', content, re.DOTALL)
        if reflection_match:
            sections['reflection'] = reflection_match.group(1).strip()
        
        return sections
    
    def process_all_weekly_targets(self, directory: str) -> List[Dict]:
        """
        Process all weekly target documents in a directory
        """
        all_chunks = []
        directory_path = Path(directory)
        
        # Find all markdown files that look like weekly targets
        files = list(directory_path.glob("**/*.md"))
        target_files = [f for f in files if 'weekly' in f.name.lower() or 'target' in f.name.lower()]
        
        if not target_files:
            # Fall back to all markdown files
            target_files = files
        
        print(f"Found {len(target_files)} weekly target documents")
        
        for file_path in sorted(target_files):
            print(f"Processing: {file_path.name}")
            
            try:
                chunks = self.process_craft_weekly_targets(str(file_path))
                all_chunks.extend(chunks)
                
                # Show stats for this week
                if chunks and chunks[0]['metadata'].get('week_date'):
                    meta = chunks[0]['metadata']
                    print(f"  Week: {meta['week_date']}")
                    print(f"  Tasks: {meta['completed_tasks']}/{meta['total_tasks']} " +
                          f"({meta['completion_rate']}%)")
            except Exception as e:
                print(f"  Error: {e}")
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
    
    def process_craft_weekly_targets(self, file_path: str) -> List[Dict]:
        """
        Process Craft weekly targets with specific format:
        # ::DD/MM/YY: Weekly Targets::
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract date from title
        date_match = re.search(r'#\s*::(\d{2}/\d{2}/\d{2}):\s*Weekly Targets::', content)
        week_date = None
        if date_match:
            # Convert DD/MM/YY to YYYY-MM-DD for consistency
            date_str = date_match.group(1)
            try:
                dt = datetime.strptime(date_str, '%d/%m/%y')
                week_date = dt.strftime('%Y-%m-%d')
            except:
                week_date = date_str
        
        # Clean Craft-specific syntax
        cleaned_content = self.clean_craft_syntax(content)
        
        # Extract tasks and metadata
        tasks_data = self.extract_weekly_tasks(content)
        
        # Build comprehensive metadata
        metadata = {
            'source': 'craft',
            'doc_type': 'weekly_targets',
            'filename': Path(file_path).name,
            'filepath': str(file_path),
            'week_date': week_date,
            'total_tasks': tasks_data['total_tasks'],
            'completed_tasks': tasks_data['completed_tasks'],
            'completion_rate': tasks_data['completion_rate'],
            'high_priority_tasks': tasks_data['high_priority_count'],
            'has_reflection': tasks_data['has_reflection']
        }
        
        # Create chunks
        # Chunk 1: Tasks list
        tasks_section = self.extract_section(content, start='# ::', end='**::Low Priority::')
        if tasks_section:
            chunks = [{
                'text': self.clean_craft_syntax(tasks_section),
                'metadata': {**metadata, 'section': 'main_tasks'}
            }]
        else:
            chunks = []
        
        # Chunk 2: Low priority tasks if present
        low_priority = self.extract_section(content, start='**::Low Priority::', end='### ::Challenge::')
        if low_priority:
            chunks.append({
                'text': self.clean_craft_syntax(low_priority),
                'metadata': {**metadata, 'section': 'low_priority'}
            })
        
        # Chunk 3: Reflection sections
        reflection_sections = self.extract_reflection_sections(content)
        for section_name, section_content in reflection_sections.items():
            if section_content.strip():
                chunks.append({
                    'text': self.clean_craft_syntax(section_content),
                    'metadata': {**metadata, 'section': section_name}
                })
        
        # If no structured chunks, fall back to full document
        if not chunks:
            chunks = [{
                'text': cleaned_content,
                'metadata': metadata
            }]
        
        return chunks
    
    def clean_craft_syntax(self, text: str) -> str:
        """Remove Craft-specific syntax like ::text::"""
        # Remove highlight markers
        text = re.sub(r'::([^:]+)::', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def extract_weekly_tasks(self, content: str) -> Dict:
        """
        Extract task statistics from weekly targets
        """
        # Find all tasks (including low priority)
        task_pattern = r'^[-*]\s*\[([ x])\]\s*(.+)$'
        tasks = []
        
        for match in re.finditer(task_pattern, content, re.MULTILINE):
            completed = match.group(1).lower() == 'x'
            task_text = match.group(2).strip()
            
            # Check if it's highlighted/high priority
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
        
        # Check if reflection is filled out
        has_reflection = bool(re.search(r'### ::Reflection::\s*\n+\w+', content))
        
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'completion_rate': round((completed / total * 100) if total > 0 else 0, 1),
            'high_priority_count': highlighted,
            'has_reflection': has_reflection,
            'tasks': tasks
        }
    
    def extract_section(self, content: str, start: str, end: str = None) -> str:
        """Extract content between two markers"""
        start_idx = content.find(start)
        if start_idx == -1:
            return ""
        
        if end:
            end_idx = content.find(end, start_idx)
            if end_idx == -1:
                return content[start_idx:]
            return content[start_idx:end_idx]
        else:
            return content[start_idx:]
    
    def extract_reflection_sections(self, content: str) -> Dict[str, str]:
        """
        Extract Challenge, Result, and Reflection sections
        """
        sections = {}
        
        # Challenge section
        challenge = self.extract_section(content, '### ::Challenge::', '### ::Result::')
        if challenge:
            sections['challenge'] = challenge.replace('### ::Challenge::', '').strip()
        
        # Result section
        result = self.extract_section(content, '### ::Result::', '### ::Reflection::')
        if result:
            sections['result'] = result.replace('### ::Result::', '').strip()
        
        # Reflection section
        reflection_match = re.search(r'### ::Reflection::\s*(.+)', content, re.DOTALL)
        if reflection_match:
            sections['reflection'] = reflection_match.group(1).strip()
        
        return sections
    
    def process_all_weekly_targets(self, directory: str) -> List[Dict]:
        """
        Process all weekly target documents in a directory
        """
        all_chunks = []
        directory_path = Path(directory)
        
        # Find all markdown files that look like weekly targets
        files = list(directory_path.glob("**/*.md"))
        target_files = [f for f in files if 'weekly' in f.name.lower() or 'target' in f.name.lower()]
        
        if not target_files:
            # Fall back to all markdown files
            target_files = files
        
        print(f"Found {len(target_files)} weekly target documents")
        
        for file_path in sorted(target_files):
            print(f"Processing: {file_path.name}")
            
            try:
                chunks = self.process_craft_weekly_targets(str(file_path))
                all_chunks.extend(chunks)
                
                # Show stats for this week
                if chunks and chunks[0]['metadata'].get('week_date'):
                    meta = chunks[0]['metadata']
                    print(f"  Week: {meta['week_date']}")
                    print(f"  Tasks: {meta['completed_tasks']}/{meta['total_tasks']} " +
                          f"({meta['completion_rate']}%)")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\nProcessed {len(all_chunks)} total chunks from weekly targets")
        return all_chunks
 
    