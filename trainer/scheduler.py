import schedule
import subprocess
import time
import socket

if __name__ == '__main__':
    def has_internet_connections(host="8.8.8.8", port=53, timeout=3):
        """
        Host: 8.8.8.8 (google-public-dns-a.google.com)
        OpenPort: 53/tcp
        Service: domain (DNS/TCP)
        """
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error as ex:
            return False

    def run():
        if not has_internet_connections():
            print("Waiting for internet connection... Checking interval: 10s Max waiting time: 2 hours.")
            slept = 0
            while not has_internet_connections():
                time.sleep(10)
                slept += 10
                if slept > 3600 * 2:
                    raise Exception("No internet connection")

        subprocess.call("python -m trainer run", shell=True)

    run()
    print("Running schedule... every day at 02:30")
    schedule.every().day.at('02:30').do(run)
    while 1:
        schedule.run_pending()
        time.sleep(60)
