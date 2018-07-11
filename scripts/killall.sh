#!/bin/bash

if pidof -x "rviz" >/dev/null; then
	kill `pidof -x "rviz"` 1>/dev/null 2>&1
	echo ""
	echo "RVIZ KILLED"
	echo "======="
else
	echo ""
	echo "RVIZ NOT RUNNING"
	echo "======="
fi

if pidof -x "roscore" >/dev/null; then
	kill `pidof -x "roscore"` 1>/dev/null 2>&1
	echo ""
	echo "ROSCORE KILLED"
	echo "======="
else
	echo ""
	echo "ROSCORE NOT RUNNING"
	echo "======="
fi