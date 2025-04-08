import os
import sys
from datetime import datetime, timedelta
import atexit
import shutil

class Documenter:
    """ Class that makes network runs self-documenting. All output data including the saved
    model, log file, parameter file and plots are saved into an output folder. """

    def __init__(self, run_name, existing_run=None, read_only=False):
        """ If existing_run is None, a new output folder named as run_name prefixed by date
        and time is created. stdout and stderr are redirected into a log file. The method
        close is registered to be automatically called when the program exits. """
        self.run_name = run_name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if existing_run is None:
            now = datetime.now()
            while True:
                full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
                # Ayo: Update this to use `out_dir` from the yaml config?
                self.basedir = os.path.join(script_dir, "../results", full_run_name)
                try:
                    os.mkdir(self.basedir)
                    break
                except FileExistsError:
                    now += timedelta(seconds=1)
        else:
            self.basedir = existing_run
            #self.basedir = os.pathi.join(script_dir, "../results", existing_run)

        if not read_only:
            self.tee = Tee(self.add_file("log.txt", False))
            atexit.register(self.close)

    def add_file(self, name, add_run_name=True):
        """ Returns the path in the output folder for a file with the given name. If
        add_run_name is True, the run name is appended to the file name. If a file with
        the same name already exists in the output folder, it is moved to a subfolder 'old'.
        """
        new_file = self.get_file(name, add_run_name)
        old_dir = os.path.join(self.basedir, "old")
        if os.path.exists(new_file):
            os.makedirs(old_dir, exist_ok=True)
            shutil.move(new_file, os.path.join(old_dir, os.path.basename(new_file)))
        return new_file

    def get_file(self, name, add_run_name=False):
        """ Returns the path in the output folder for a file with the given name. If
        add_run_name is True, the run name is appended to the file name. """
        if add_run_name:
            name_base, name_ext = os.path.splitext(name)
            name = f"{name_base}_{self.run_name}{name_ext}"
        return os.path.join(self.basedir,name)

    def close(self):
        """ Ends redirection of stdout and changes the file permissions of the output folder
        such that other people on the cluster can access the files. """
        self.tee.close()
        os.system("chmod -R 755 " + self.basedir)

class Tee(object):
    """ Class to replace stdout and stderr. It redirects all printed data to std_out as well
    as a log file. """

    def __init__(self, log_file):
        """ Creates log file and redirects stdout and stderr. """
        self.log_file = open(log_file, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def close(self):
        """ Closes log file and restores stdout and stderr. """
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    def write(self, data):
        """ Writes data to stdout and the log file. """
        self.log_file.write(data)
        self.stdout.write(data)

    def flush(self):
        """ Flushes buffered data to the file. """
        self.log_file.flush()
