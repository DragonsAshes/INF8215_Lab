#!/bin/bash
PORT=6666

trap finish INT

function finish() {
	echo "GOODBYE"
	kill -9 $PID
}

python my_player.py -b localhost -p $PORT &
PID=$!
python game.py http://localhost:$PORT human --time 300

finish
