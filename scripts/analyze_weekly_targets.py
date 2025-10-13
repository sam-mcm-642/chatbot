"""
Analyze your weekly targets for insights
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.document_processor import DocumentProcessor
from collections import Counter
from datetime import datetime
import json

def analyze_targets(directory: str = "data/raw/craft_export/weekly_targets"):
    """
    Analyze completion rates and patterns in weekly targets
    """
    processor = DocumentProcessor()
    
    print("="*60)
    print("Weekly Targets Analysis")
    print("="*60)
    
    # Process all weekly targets
    chunks = processor.process_all_weekly_targets(directory)
    
    if not chunks:
        print("No weekly target documents found!")
        return
    
    # Extract all weeks' metadata
    weeks = {}
    all_tasks = []
    
    for chunk in chunks:
        meta = chunk['metadata']
        week_date = meta.get('week_date')
        
        if week_date and meta.get('section') == 'main_tasks':
            weeks[week_date] = {
                'total': meta['total_tasks'],
                'completed': meta['completed_tasks'],
                'rate': meta['completion_rate']
            }
        
        # Extract individual tasks for pattern analysis
        if meta.get('section') == 'main_tasks':
            text = chunk['text']
            for line in text.split('\n'):
                if line.strip().startswith('- ['):
                    completed = '[x]' in line or '[X]' in line
                    task = line.split(']', 1)[1].strip() if ']' in line else line
                    all_tasks.append({
                        'text': task,
                        'completed': completed,
                        'week': week_date
                    })
    
    # Overall statistics
    print(f"\n📊 Overall Statistics")
    print(f"{'='*60}")
    print(f"Total weeks tracked: {len(weeks)}")
    
    if weeks:
        avg_completion = sum(w['rate'] for w in weeks.values()) / len(weeks)
        total_tasks = sum(w['total'] for w in weeks.values())
        total_completed = sum(w['completed'] for w in weeks.values())
        
        print(f"Average completion rate: {avg_completion:.1f}%")
        print(f"Total tasks set: {total_tasks}")
        print(f"Total completed: {total_completed}")
        
        # Best and worst weeks
        best_week = max(weeks.items(), key=lambda x: x[1]['rate'])
        worst_week = min(weeks.items(), key=lambda x: x[1]['rate'])
        
        print(f"\n🎯 Best week: {best_week[0]} ({best_week[1]['rate']:.1f}%)")
        print(f"📉 Most challenging week: {worst_week[0]} ({worst_week[1]['rate']:.1f}%)")
    
    # Recurring tasks analysis
    print(f"\n🔄 Recurring Goals")
    print(f"{'='*60}")
    
    # Normalize task text for comparison
    task_texts = [t['text'].lower().strip() for t in all_tasks]
    common_tasks = Counter(task_texts).most_common(10)
    
    for task, count in common_tasks:
        if count > 1:  # Only show recurring
            completed_count = sum(1 for t in all_tasks 
                                 if t['text'].lower().strip() == task and t['completed'])
            completion_rate = (completed_count / count * 100) if count > 0 else 0
            print(f"  {count}x - {task[:50]}")
            print(f"       Completed: {completed_count}/{count} ({completion_rate:.0f}%)")
    
    # Task categories (simple keyword matching)
    print(f"\n📋 Task Categories")
    print(f"{'='*60}")
    
    categories = {
        'Job Search': ['job', 'cv', 'linkedin', 'apply'],
        'Fitness': ['gym', 'training', 'exercise', 'creatine'],
        'Projects': ['project', 'gaa jersey', 'ai'],
        'Admin': ['budget', 'organised', 'sorted', 'cleaned'],
        'Personal Care': ['haircut', 'clothes']
    }
    
    for category, keywords in categories.items():
        category_tasks = [t for t in all_tasks 
                         if any(kw in t['text'].lower() for kw in keywords)]
        if category_tasks:
            completed = sum(1 for t in category_tasks if t['completed'])
            total = len(category_tasks)
            rate = (completed / total * 100) if total > 0 else 0
            print(f"  {category}: {completed}/{total} ({rate:.0f}%)")
    
    # Save detailed analysis
    output = {
        'summary': {
            'weeks_tracked': len(weeks),
            'avg_completion_rate': avg_completion if weeks else 0,
            'total_tasks': total_tasks if weeks else 0
        },
        'by_week': weeks,
        'recurring_tasks': dict(common_tasks)
    }
    
    output_path = Path("data/processed/weekly_targets_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to {output_path}")
    print("="*60)

if __name__ == "__main__":
    analyze_targets()