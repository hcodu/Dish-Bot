@echo off
REM Batch script to connect to remote machine and kill all Python processes

echo Connecting to jp@192.168.1.242 and killing all Python processes...

REM Using plink (PuTTY command line tool) with password
REM If you don't have plink, install PuTTY or use the sshpass alternative below
plink -ssh jp@192.168.1.242 -pw racks123 "pkill -9 python; pkill -9 python3"

REM Alternative using OpenSSH (if sshpass is installed):
REM sshpass -p racks123 ssh -o StrictHostKeyChecking=no jp@192.168.1.242 "pkill -9 python; pkill -9 python3"

REM Alternative using standard ssh (you'll need to enter password manually):
REM ssh jp@192.168.1.242 "pkill -9 python; pkill -9 python3"

echo Done!
pause
