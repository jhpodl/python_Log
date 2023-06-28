from tkinter import ttk

import paramiko
import sys
import tkinter as tk
import tkinter.messagebox as messagebox

sys.path.insert(0, "/")
from util.xutil import *


def egrepclosure(*exprs):
    ptcs = [re.compile(expr) for expr in exprs]

    def egrep(lines):
        for line in lines:
            if all(ptc.search(line) for ptc in ptcs):
                yield line

    return egrep


def text_to_dict(lines):
    for line in lines:
        yield orjson.loads(line)


def extract_to_dict(dicts, *keys):
    for dict in dicts:
        val_list = []
        for key in keys:
            val_list.append(dict[key])

        yield val_list


def copy_grep_results(filename, entry):
    search_terms = entry.get().split()
    egrep = egrepclosure(*search_terms)

    file_stream = file_path_stream('C:\\Users\\admin\\Documents\\', filename)
    lines = line_stream(file_stream)

    lines_list = [line + '\n' for line in egrep(lines)]

    all_lines = ''.join(lines_list)

    root.clipboard_clear()
    root.clipboard_append(all_lines)
    messagebox.showinfo("copy_grep_results completed", "copy_grep_results completed")


current_date = datetime.now().strftime("%Y-%m-%d")


def getLogDev():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('10.180.10.50', port=22, username='crdhkim', password='C0mpassion23$')

    stdin, stdout, stderr = ssh.exec_command("ls /data/logs/catalina/")
    file_list = stdout.read().decode().splitlines()

    ssh.close()
    return file_list


def getLogProd(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if server == "prod1":
        ssh.connect('10.180.10.61', port=22, username='prodwas1', password='ZjaVoTus12#$')
    elif server == "prod2":
        ssh.connect('10.180.10.62', port=22, username='prodwas2', password='ZjaVoTus12#$')

    stdin, stdout, stderr = ssh.exec_command("ls /data/logs/catalina/")
    file_list = stdout.read().decode().splitlines()

    ssh.close()
    return file_list


def on_get_file_button_clicked(server):
    if server == "dev":
        options = getLogDev()
    elif server.__contains__("prod"):
        options = getLogProd(server)

    combo = ttk.Combobox(root, values=options)
    combo.pack()
    widgets.append(combo)

    download_button = tk.Button(root, text="Download selected file", command=lambda: download_file(combo.get(), server))
    download_button.pack()
    widgets.append(download_button)

    message_label = tk.Label(root, text="Please enter a search word above entry")
    message_label.pack(side=tk.TOP)
    widgets.append(message_label)

    entry = tk.Entry(root, width=50)
    entry.pack(side=tk.TOP)
    widgets.append(entry)

    button = tk.Button(root, text="SearchAndCopy", command=lambda: copy_grep_results(combo.get(), entry),
                       state='disable')
    button.pack(anchor="center")
    widgets.append(button)

def download_file(filename, server):
    for widget in root.winfo_children():
        widget.configure(state='disable')
    root.config(cursor='wait')
    root.update()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if server == "dev":
        ssh.connect('10.180.10.50', port=22, username='crdhkim', password='C0mpassion23$')
    elif server == "prod1":
        ssh.connect('10.180.10.61', port=22, username='prodwas1', password='ZjaVoTus12#$')
    elif server == "prod2":
        ssh.connect('10.180.10.62', port=22, username='prodwas2', password='ZjaVoTus12#$')

    sftp = ssh.open_sftp()
    remoteFilename = f"/data/logs/catalina/{filename}"
    localFilename = f"C:\\Users\\admin\\Documents\\{filename}"
    sftp.get(remoteFilename, localFilename)

    sftp.close()
    ssh.close()

    for widget in root.winfo_children():
        widget.configure(state='normal')
    root.config(cursor='')
    root.update()

    messagebox.showinfo("File transfer completed", f"File {localFilename} has been successfully downloaded")


def on_get_log_button_clicked():
    server = combo.get()
    for widget in widgets:
        if widget not in (combo, get_log_button):
            widget.destroy()
    on_get_file_button_clicked(server)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Compassion Log v0.1")

    widgets = []

    combo = ttk.Combobox(root, values=["dev", "prod1", "prod2"])
    combo.pack(side=tk.TOP)
    widgets.append(combo)

    get_log_button = tk.Button(root, text="Get Log", command=on_get_log_button_clicked)
    get_log_button.pack(side=tk.TOP)
    widgets.append(get_log_button)

    root.mainloop()
