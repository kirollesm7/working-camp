@echo off
rem Opens the clothes counter numbers screen full screen (the server must be running)
start "" msedge --new-window --start-fullscreen --app=http://localhost:5000/display 2>nul || start "" http://localhost:5000/display
