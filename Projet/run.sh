#!/bin/bash
PORT1=6668
PORT2=7779

trap finish INT

function finish() {
	echo "GOODBYE"
	kill -9 $PID1
	kill -9 $PID2
}

python my_player.py -b localhost -p $PORT1 &
PID1=$!
python my_old_player.py -b localhost -p $PORT2 &
PID2=$!
python game.py http://localhost:$PORT1 http://localhost:$PORT2 --time 3000
#python game.py http://localhost:$PORT1 human --time 3000
#python game.py human human --time 3000

finish
