@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  GitHub 推送：令牌输入时不显示，不会留痕
echo ============================================
".venv\Scripts\python.exe" "scripts\12_push_github.py" > "push_log.txt" 2>&1
type "push_log.txt"
echo.
pause
