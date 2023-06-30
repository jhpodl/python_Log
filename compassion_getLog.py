import configparser
from tkinter import ttk

import webbrowser
import paramiko
import sys
import tkinter as tk
import tkinter.messagebox as messagebox

sys.path.insert(0, "/")
from util.xutil import *

current_date = datetime.now().strftime("%Y-%m-%d")


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

    lines_list = [line + '\n' for line in egrep(lines)]
    show_result(lines_list)


def get_server_filelist(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(config.get(server, 'ip'), port=int(config.get(server, 'port')), username=config.get(server, 'username'),
                password=config.get(server, 'password'))

    stdin, stdout, stderr = ssh.exec_command(config.get("COMMAND", 'ls_command'))
    file_list = stdout.read().decode().splitlines()

    ssh.close()
    return file_list


def on_get_file_button_clicked(server):
    server_list_combo = add_widget(widgets_init_download, ttk.Combobox, values=get_server_filelist(server))
    download_button = add_widget(widgets_init_download, tk.Button, text=config.get("MESSAGE", 'download'),
                                 command=lambda: download_file(server_list_combo, server.upper()))


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
    widget_destroy(widgets_init_download)
    widget_lock("disable", "wait")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(config.get(server, 'ip'), port=int(config.get(server, 'port')), username=config.get(server, 'username'),
                password=config.get(server, 'password'))

    sftp = ssh.open_sftp()

    log_filename=server_list_combo.get()
    sftp.get(config.get('LOG_PATHS', 'origin_log_path') + log_filename,
             config.get('LOG_PATHS', 'source_log_path') + log_filename)

    sftp.close()
    ssh.close()

    widget_lock("normal", "")

    message_label = add_widget("", tk.Label, text=config.get('MESSAGE', 'search_text'))
    entry = add_widget("", tk.Entry)
    button = add_widget("", tk.Button, text="SearchAndCopy", command=lambda: copy_grep_results(
        line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename)), entry))
    message_label = add_widget("", tk.Label, text="ID list")

    id_list = make_id_list(line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename)))
    id_combo = add_widget("", ttk.Combobox, values=list(id_list))
    id_combo.set(config.get('MESSAGE', 'combo_search'))

    def search_ids(event):
        search_text = id_combo.get()
        filtered_ids = [id for id in id_list if search_text.lower() in id.lower()]
        id_combo['values'] = filtered_ids

    id_combo.bind('<KeyRelease>', search_ids)

    def on_combobox_changed(event):
        widget_lock("disable", "wait")
        widget_destroy(widgets_combo)

        selected_item = id_combo.get()
        print("Selected item is:", selected_item)

        egrep = egrepclosure(selected_item)
        lines_list = [line + '\n' for line in
                      egrep(line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename)))]

        session_label = add_widget("", tk.Label, text="Session List")
        session_list = set([])
        for line in lines_list:
            matches = re.findall(r"\[(.*?)]", line)
            if matches:
                try:
                    datetime.strptime(matches[0], "%Y-%m-%d %H:%M:%S")
                    session_list.add(matches[1])
                except ValueError:
                    print(f"{matches}")

        session_combo = add_widget("", ttk.Combobox, values=list(session_list))

        def on_session_combo_changed(event):
            widget_lock("disable", "wait")
            selected_item = session_combo.get()
            print("Selected item is:", selected_item)

            egrep = egrepclosure(selected_item)
            err_str = []
            lines_list = []
            for line in egrep(line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename))):
                line = line + '\n'
                if "ERROR" in line:
                    err_str.append(line)
                else:
                    lines_list.append(line)
            lines_list = err_str + lines_list
            show_result(lines_list)
            widget_lock("normal", "")

        session_combo.bind("<<ComboboxSelected>>", on_session_combo_changed)
        show_result(lines_list)
        widget_lock("normal", "")

    id_combo.bind("<<ComboboxSelected>>", on_combobox_changed)
    widgets_combo = widgets_init_download.copy()
    widgets_combo.append(id_combo)


def show_result(lines_list):
    all_lines = ''.join(lines_list)
    result_filename = config.get('LOG_PATHS', 'result')
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(all_lines)
    webbrowser.open(result_filename)
    root.clipboard_clear()
    root.clipboard_append(all_lines)


def widget_destroy(list):
    for widget in widgets:
        if widget not in list:
            widget.destroy()


def widget_lock(state, cursor):
    for widget in root.winfo_children():
        widget.configure(state=state)
    root.config(cursor=cursor)
    root.update()


def on_get_log_button_clicked():
    server = combo.get()
    widget_destroy((combo, get_log_button))
    on_get_file_button_clicked(server.upper())


def add_widget(widget_list, widget_type, *args, **kwargs):
    widget = widget_type(root, *args, **kwargs)
    widget.pack(side=tk.TOP)
    widgets.append(widget)
    if widget_list != "":
        widget_list.append(widget)
    return widget


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Compassion Log v0.2")

    widgets = []
    widgets_init_download = []
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    combo = add_widget(widgets_init_download, ttk.Combobox, values=["dev", "prod1", "prod2"])
    get_log_button = add_widget(widgets_init_download, tk.Button, text="Get Log", command=on_get_log_button_clicked)

    root.mainloop()
