from tkinter import ttk

import webbrowser
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


# 라이브러리
def text_to_dict(lines):
    for line in lines:
        yield orjson.loads(line)


def extract_to_dict(dicts, *keys):
    for dict in dicts:
        val_list = []
        for key in keys:
            val_list.append(dict[key])

        yield val_list


def copy_grep_results(lines, entry):
    search_terms = entry.get().split()
    egrep = egrepclosure(*search_terms)

    # error = egrepclosure("ERROR") , userId로 select할때 그 해당하는 유저의 에러로그만 뽑아주도록

    # file_stream = file_path_stream('C:\\Users\\admin\\Documents\\', filename)
    # lines = line_stream(file_stream)

    lines_list = [line + '\n' for line in egrep(lines)]

    all_lines = ''.join(lines_list)

    result_filename = 'C:\\Users\\admin\\Documents\\result.txt'
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(all_lines)

    webbrowser.open(result_filename)

    root.clipboard_clear()
    root.clipboard_append(all_lines)

    # messagebox.showinfo("copy_grep_results completed", "copy_grep_results completed")


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

    server_list_combo = ttk.Combobox(root, values=options)
    server_list_combo.pack()
    widgets.append(server_list_combo)
    widgets_init_download.append(server_list_combo)

    download_button = tk.Button(root, text="Download selected file",
                                command=lambda: download_file(server_list_combo, server))
    download_button.pack()
    widgets.append(download_button)
    widgets_init_download.append(download_button)


def make_id_list(lines):
    username_egrep = egrepclosure("userId")
    lines_list = set([])
    for line in username_egrep(lines):
        match = re.search(r'"userId":"(.*?)"', line)
        if match:
            user_id = match.group(1)  # extract the userId
            lines_list.add(user_id)
    return lines_list


def download_file(server_list_combo, server):
    for widget in widgets:
        if widget not in widgets_init_download:
            widget.destroy()

    widget_control("disable", "wait")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if server == "dev":
        ssh.connect('10.180.10.50', port=22, username='crdhkim', password='C0mpassion23$')
    elif server == "prod1":
        ssh.connect('10.180.10.61', port=22, username='prodwas1', password='ZjaVoTus12#$')
    elif server == "prod2":
        ssh.connect('10.180.10.62', port=22, username='prodwas2', password='ZjaVoTus12#$')

    sftp = ssh.open_sftp()
    remoteFilename = f"/data/logs/catalina/{server_list_combo.get()}"
    localFilename = f"C:\\Users\\admin\\Documents\\{server_list_combo.get()}"
    sftp.get(remoteFilename, localFilename)

    sftp.close()
    ssh.close()

    widget_control("normal", "")

    message_label = tk.Label(root, text="Please enter a search word above entry")
    message_label.pack(side=tk.TOP)
    widgets.append(message_label)

    entry = tk.Entry(root, width=50)
    entry.pack(side=tk.TOP)
    widgets.append(entry)

    button = tk.Button(root, text="SearchAndCopy", command=lambda: copy_grep_results(
        line_stream(file_path_stream('C:\\Users\\admin\\Documents\\', server_list_combo.get())), entry))
    # ,state='disabled')
    button.pack(anchor="center")
    widgets.append(button)

    message_label = tk.Label(root, text="ID list")
    message_label.pack(side=tk.TOP)
    widgets.append(message_label)

    placeholder_text = "Search an item..."
    id_list = make_id_list(line_stream(file_path_stream('C:\\Users\\admin\\Documents\\', server_list_combo.get())))
    id_combo = ttk.Combobox(root, values=list(id_list))
    id_combo.set(placeholder_text)

    def search_ids(event):
        # Get the current text in the combobox
        search_text = id_combo.get()
        # Filter the ids based on the search text
        filtered_ids = [id for id in id_list if search_text.lower() in id.lower()]
        # Update the values in the combobox with the filtered ids
        id_combo['values'] = filtered_ids

    id_combo.pack()
    id_combo.bind('<KeyRelease>', search_ids)

    def on_combobox_changed(event):
        widget_control("disable", "wait")
        for widget in widgets:
            if widget not in widgets_combo:
                widget.destroy()

        selected_item = id_combo.get()
        print("Selected item is:", selected_item)

        egrep = egrepclosure(selected_item)
        lines_list = [line + '\n' for line in
                      egrep(line_stream(file_path_stream('C:\\Users\\admin\\Documents\\', server_list_combo.get())))]

        session_label = tk.Label(root, text="Session List")
        session_label.pack(side=tk.TOP)
        widgets.append(session_label)

        session_list = set([])
        for line in lines_list:
            matches = re.findall(r"\[(.*?)\]", line)
            if matches:
                try:
                    datetime.strptime(matches[0], "%Y-%m-%d %H:%M:%S")
                    session_list.add(matches[1])
                except ValueError:
                    print(f"{matches}")

        session_combo = ttk.Combobox(root, values=list(session_list))

        def on_session_combo_changed(event):
            widget_control("disable", "wait")
            selected_item = session_combo.get()
            print("Selected item is:", selected_item)

            egrep = egrepclosure(selected_item)
            err_str = []
            lines_list = []
            for line in egrep(line_stream(file_path_stream('C:\\Users\\admin\\Documents\\', server_list_combo.get()))):
                line = line + '\n'
                if "ERROR" in line:
                    err_str.append(line)
                else:
                    lines_list.append(line)
            lines_list = err_str + lines_list
            all_lines = ''.join(lines_list)
            result_filename = 'C:\\Users\\admin\\Documents\\result.txt'
            with open(result_filename, 'w', encoding='utf-8') as f:
                f.write(all_lines)
            webbrowser.open(result_filename)
            root.clipboard_clear()
            root.clipboard_append(all_lines)
            widget_control("normal", "")

        session_combo.bind("<<ComboboxSelected>>", on_session_combo_changed)
        session_combo.pack()
        widgets.append(session_combo)

        all_lines = ''.join(lines_list)

        result_filename = 'C:\\Users\\admin\\Documents\\result.txt'
        with open(result_filename, 'w', encoding='utf-8') as f:
            f.write(all_lines)

        webbrowser.open(result_filename)
        root.clipboard_clear()
        root.clipboard_append(all_lines)
        widget_control("normal", "")

        # button = tk.Button(root, text=selected_item+"'s ERROR log", command=lambda: get_Error_logs(
        #     line_stream(file_path_stream('C:\\Users\\admin\\Documents\\', server_list_combo.get())), entry))
        # # ,state='disabled')
        # button.pack(anchor="center")
        # widgets.append(button)

    # def get_Error_logs():

    id_combo.bind("<<ComboboxSelected>>", on_combobox_changed)
    widgets.append(id_combo)
    widgets_combo=widgets_init_download.copy()
    widgets_combo.append(id_combo)

    #widgets_init_download.append(id_combo)
    #messagebox.showinfo("File transfer completed", f"File {localFilename} has been successfully downloaded")


def widget_control(state, cursor):
    for widget in root.winfo_children():
        widget.configure(state=state)
    root.config(cursor=cursor)
    root.update()


def on_get_log_button_clicked():
    server = combo.get()
    for widget in widgets:
        if widget not in (combo, get_log_button):
            widget.destroy()
    on_get_file_button_clicked(server)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Compassion Log v0.2")

    widgets = []
    widgets_init_download = []

    combo = ttk.Combobox(root, values=["dev", "prod1", "prod2"])
    combo.pack(side=tk.TOP)
    widgets.append(combo)
    widgets_init_download.append(combo)

    get_log_button = tk.Button(root, text="Get Log", command=on_get_log_button_clicked)
    get_log_button.pack(side=tk.TOP)
    widgets.append(get_log_button)
    widgets_init_download.append(get_log_button)

    root.mainloop()
