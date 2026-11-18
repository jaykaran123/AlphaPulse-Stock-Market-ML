@echo off
title Launching Stock Market ML CA1 & CA2 Dashboard
echo =========================================================================
echo    Stock Market ML CA1 & CA2 Quantitative Dashboard
echo    Multi-Algorithm Price Predictor & Volatility Regime Segmentation
echo =========================================================================
echo.
echo Launching Interactive Dashboard in your default web browser...
start "" "%~dp0web_interface\index.html"
echo.
echo Dashboard successfully launched!
echo You can also start the local Node or Python server if desired:
echo   - Node.js server: cd web_interface ^&^& node server.js
echo   - Python server:  cd web_interface ^&^& python server.py
echo.
pause
