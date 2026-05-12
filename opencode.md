# AI Agent Rules

## File Creation Policy
ALWAYS ask the user before creating new files.

## Rules
1. **Ask before creating** - Before creating any new .py file, ask: "Should I create a new file for this?"
2. **Prefer extending** - Suggest adding to existing files (desktop_poi_viewer.py, relabel_corners_tool.py) first
3. **No one-time scripts** - If a temporary script is needed, ask where to put it and clean up after
4. **Centralize** - All functionality should ideally be in desktop_poi_viewer.py or relabel_corners_tool.py
5. **Explain first** - Before creating anything, explain what you want to create and why
6. **User decides** - The user's answer determines whether to proceed with file creation

## Project Structure
- `desktop_poi_viewer.py` - Main GUI (Viewer + Relabel + Train tabs)
- `relabel_corners_tool.py` - Relabeling functionality
- `scripts/` - ONLY reusable, documented utility scripts (if needed)

## Before Creating Any File
1. Does this already exist?
2. Can it be added to an existing file?
3. Have I asked the user for permission?

If "no" to any: STOP and ASK.
