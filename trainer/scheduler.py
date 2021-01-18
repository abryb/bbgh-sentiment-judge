import schedule
import subprocess
import time

if __name__ == '__main__':
    def run():
        subprocess.call("python -m trainer run", shell=True)

    run()
    print("Running schedule... every day at 02:30")
    schedule.every().day.at('02:30').do(run)
    while 1:
        schedule.run_pending()
        time.sleep(60)