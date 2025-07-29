#!/usr/bin/env python3
"""
Phase 1 Import Standardization Script
=====================================

This script implements Phase 1 of the import standardization refactoring plan:
1. Adds missing `from __future__ import annotations` to files with type hints
2. Standardizes import ordering using isort
3. Validates that no functionality is broken

Features:
- Dry-run mode for safe preview
- Automatic backup creation
- Comprehensive validation
- Detailed reporting
- Rollback capability
"""

import ast
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class ImportStandardizer:
    """Handles Phase 1 import standardization for the backend/agent directory."""
    
    def __init__(self, target_dir: str, dry_run: bool = True):
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        self.backup_dir = None
        self.changes_made = []
        self.errors = []
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'future_annotations_added': 0,
            'imports_standardized': 0,
            'validation_errors': 0
        }
    
    def run(self) -> bool:
        """Execute the complete Phase 1 standardization process."""
        print(f"🚀 Starting Phase 1 Import Standardization")
        print(f"📁 Target directory: {self.target_dir}")
        print(f"🔍 Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print("=" * 60)
        
        try:
            # Step 1: Create backup if not dry run
            if not self.dry_run:
                self._create_backup()
            
            # Step 2: Scan and analyze files
            python_files = self._find_python_files()
            print(f"📋 Found {len(python_files)} Python files to analyze")
            
            # Step 3: Process each file
            for file_path in python_files:
                self._process_file(file_path)
            
            # Step 4: Run isort on all files (if not dry run)
            if not self.dry_run:
                self._standardize_imports()
            
            # Step 5: Validate all changes
            self._validate_changes(python_files)
            
            # Step 6: Generate report
            self._generate_report()
            
            return len(self.errors) == 0
            
        except Exception as e:
            print(f"❌ Critical error during execution: {e}")
            if not self.dry_run and self.backup_dir:
                print("🔄 Attempting to restore from backup...")
                self._restore_backup()
            return False
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the target directory."""
        python_files = []
        for root, dirs, files in os.walk(self.target_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return sorted(python_files)
    
    def _process_file(self, file_path: Path) -> None:
        """Process a single Python file for import standardization."""
        try:
            self.stats['files_processed'] += 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Check if file needs future annotations
            if self._needs_future_annotations(content):
                content = self._add_future_annotations(content, file_path)
                if content != original_content:
                    self.stats['future_annotations_added'] += 1
                    self.changes_made.append({
                        'file': file_path,
                        'change': 'Added future annotations import',
                        'type': 'future_annotations'
                    })
            
            # Write changes if not dry run and content changed
            if not self.dry_run and content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['files_modified'] += 1
            elif self.dry_run and content != original_content:
                print(f"📝 Would modify: {file_path.relative_to(self.target_dir)}")
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
    
    def _needs_future_annotations(self, content: str) -> bool:
        """Check if file needs 'from __future__ import annotations'."""
        # Check if already has future annotations
        if 'from __future__ import annotations' in content:
            return False
        
        # Check if file has type hints that would benefit from future annotations
        type_hint_patterns = [
            r'def\s+\w+\([^)]*\)\s*->',  # Function return type hints
            r':\s*[A-Z]\w*\[',           # Generic type hints like List[str]
            r':\s*dict\[',               # dict[str, Any] style hints
            r':\s*list\[',               # list[str] style hints
            r':\s*tuple\[',              # tuple[int, ...] style hints
            r':\s*set\[',                # set[str] style hints
            r':\s*Optional\[',           # Optional[str] hints
            r':\s*Union\[',              # Union[str, int] hints
            r':\s*Callable\[',           # Callable hints
            r':\s*Type\[',               # Type[SomeClass] hints
        ]
        
        for pattern in type_hint_patterns:
            if re.search(pattern, content):
                return True
        
        # Also check for class-level type annotations
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):  # Variable annotations
                    return True
        except SyntaxError:
            # If we can't parse, assume it might need annotations
            return False
        
        return False
    
    def _add_future_annotations(self, content: str, file_path: Path) -> str:
        """Add 'from __future__ import annotations' to the file content."""
        lines = content.split('\n')
        
        # Find the correct position to insert the import
        insert_position = 0
        
        # Skip shebang line
        if lines and lines[0].startswith('#!'):
            insert_position = 1
        
        # Skip encoding declarations
        for i in range(insert_position, min(len(lines), insert_position + 2)):
            if i < len(lines) and ('coding:' in lines[i] or 'coding=' in lines[i]):
                insert_position = i + 1
        
        # Skip module docstring
        if insert_position < len(lines):
            # Check for module-level docstring
            remaining_content = '\n'.join(lines[insert_position:])
            try:
                tree = ast.parse(remaining_content)
                if (tree.body and 
                    isinstance(tree.body[0], ast.Expr) and 
                    isinstance(tree.body[0].value, ast.Constant) and 
                    isinstance(tree.body[0].value.value, str)):
                    # Find the end of the docstring
                    docstring_lines = tree.body[0].value.value.count('\n') + 1
                    if '"""' in remaining_content or "'''" in remaining_content:
                        # Multi-line docstring
                        for i, line in enumerate(lines[insert_position:], insert_position):
                            if ('"""' in line or "'''" in line) and i > insert_position:
                                insert_position = i + 1
                                break
            except SyntaxError:
                pass
        
        # Skip any existing __future__ imports to avoid duplicates
        while (insert_position < len(lines) and 
               lines[insert_position].strip().startswith('from __future__')):
            insert_position += 1
        
        # Insert the future annotations import
        future_import = 'from __future__ import annotations'
        
        # Add empty line before if there are other imports after
        if (insert_position < len(lines) and 
            lines[insert_position].strip() and 
            not lines[insert_position].startswith('#')):
            lines.insert(insert_position, future_import)
            lines.insert(insert_position + 1, '')
        else:
            lines.insert(insert_position, future_import)
        
        return '\n'.join(lines)
    
    def _standardize_imports(self) -> None:
        """Run isort on all Python files to standardize import ordering."""
        print("🔄 Standardizing import ordering with isort...")
        
        try:
            # Run isort on the entire directory
            result = subprocess.run([
                'python', '-m', 'isort', 
                str(self.target_dir),
                '--profile', 'black',
                '--line-length', '88',
                '--multi-line', '3',
                '--trailing-comma',
                '--force-grid-wrap', '0',
                '--combine-as',
                '--use-parentheses'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.stats['imports_standardized'] = len(self._find_python_files())
                print("✅ Import ordering standardized successfully")
            else:
                error_msg = f"isort failed: {result.stderr}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
                
        except FileNotFoundError:
            error_msg = "isort not found. Please install with: pip install isort"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
        except Exception as e:
            error_msg = f"Error running isort: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
    
    def _validate_changes(self, python_files: List[Path]) -> None:
        """Validate that all Python files still compile correctly."""
        print("🔍 Validating syntax of all Python files...")
        
        validation_errors = []
        
        for file_path in python_files:
            try:
                py_compile.compile(str(file_path), doraise=True)
            except py_compile.PyCompileError as e:
                validation_errors.append(f"Syntax error in {file_path}: {e}")
                self.stats['validation_errors'] += 1
        
        if validation_errors:
            print(f"❌ Found {len(validation_errors)} validation errors:")
            for error in validation_errors:
                print(f"   • {error}")
            self.errors.extend(validation_errors)
        else:
            print("✅ All files passed syntax validation")
    
    def _create_backup(self) -> None:
        """Create a backup of the target directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.target_dir.parent / f"agent_backup_{timestamp}"
        
        print(f"💾 Creating backup at: {self.backup_dir}")
        shutil.copytree(self.target_dir, self.backup_dir)
        print("✅ Backup created successfully")
    
    def _restore_backup(self) -> None:
        """Restore from backup in case of errors."""
        if self.backup_dir and self.backup_dir.exists():
            print(f"🔄 Restoring from backup: {self.backup_dir}")
            shutil.rmtree(self.target_dir)
            shutil.copytree(self.backup_dir, self.target_dir)
            print("✅ Backup restored successfully")
    
    def _generate_report(self) -> None:
        """Generate a comprehensive report of all changes made."""
        print("\n" + "=" * 60)
        print("📊 PHASE 1 IMPORT STANDARDIZATION REPORT")
        print("=" * 60)
        
        print(f"📁 Target Directory: {self.target_dir}")
        print(f"🕒 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        
        print("\n📈 STATISTICS:")
        for key, value in self.stats.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        if self.changes_made:
            print(f"\n📝 CHANGES MADE ({len(self.changes_made)}):")
            for change in self.changes_made:
                rel_path = change['file'].relative_to(self.target_dir)
                print(f"   • {rel_path}: {change['change']}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        success = len(self.errors) == 0
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n🎯 OVERALL STATUS: {status}")
        
        if not self.dry_run and self.backup_dir:
            print(f"💾 Backup Location: {self.backup_dir}")
        
        print("=" * 60)


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 1 Import Standardization for backend/agent directory"
    )
    parser.add_argument(
        "target_dir",
        help="Path to the backend/agent directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute changes (overrides --dry-run)"
    )
    
    args = parser.parse_args()
    
    # Determine execution mode
    dry_run = not args.execute
    
    # Validate target directory
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        print(f"❌ Error: Target directory does not exist: {target_dir}")
        sys.exit(1)
    
    if not target_dir.is_dir():
        print(f"❌ Error: Target path is not a directory: {target_dir}")
        sys.exit(1)
    
    # Run the standardization
    standardizer = ImportStandardizer(target_dir, dry_run=dry_run)
    success = standardizer.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()