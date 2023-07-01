import configparser
from tkinter import ttk

import webbrowser
import paramiko
import sys
import tkinter as tk
import tkinter.messagebox as messagebox

sys.path.insert(0, "/")
from util.xutil import *


class WidgetManager:
    def __init__(self):
        self.widgets = []
        self.widgets_init_download = []

    def add_widget(self, widget_type, *args, widget_list="", **kwargs):
        """Add a widget to the root and widget lists."""
        widget = widget_type(root, *args, **kwargs)
        widget.pack(side=tk.TOP)
        self.widgets.append(widget)
        if widget_list != "":
            widget_list.append(widget)
        return widget

    def widget_destroy(self, list):
        """Destroy all widgets except for those specified in the list."""
        for widget in self.widgets:
            if widget not in list:
                widget.destroy()

    def widget_lock(self, state, cursor):
        """Lock all widgets and set cursor to specific state."""
        for widget in root.winfo_children():
            widget.configure(state=state)
        root.config(cursor=cursor)
        root.update()

    def get_widgets(self):
        """Return list of all widgets."""
        return self.widgets


widget_manager = WidgetManager()
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
    ssh = connect_ssh(server)

    stdin, stdout, stderr = ssh.exec_command(config.get("COMMAND", 'ls_command'))
    file_list = stdout.read().decode().splitlines()

    ssh.close()
    return file_list


def make_id_list(lines):
    username_egrep = egrepclosure("userId")
    lines_list = set([])
    for line in username_egrep(lines):
        match = re.search(r'"userId":"(.*?)"', line)
        if match:
            user_id = match.group(1)  # extract the userId
            lines_list.add(user_id)
    return lines_list


def connect_ssh(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(config.get(server, 'ip'), port=int(config.get(server, 'port')),
                username=config.get(server, 'username'), password=config.get(server, 'password'))
    return ssh


def download_file(log_filename, server):
    widget_manager.widget_destroy(widget_manager.widgets_init_download)
    widget_manager.widget_lock("disable", "wait")
    ssh = connect_ssh(server)
    sftp = ssh.open_sftp()
    sftp.get(config.get('LOG_PATHS', 'origin_log_path') + log_filename,
             config.get('LOG_PATHS', 'source_log_path') + log_filename)
    sftp.close()
    ssh.close()
    widget_manager.widget_lock("normal", "")
    log_lines = line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename))
    initialize_widgets(log_filename, log_lines)


def process_selected_item(selected_item, log_filename):
    egrep = egrepclosure(selected_item)
    lines_list = [line + '\n' for line in
                  egrep(line_stream(file_path_stream(config.get('LOG_PATHS', 'source_log_path'), log_filename)))]
    return lines_list


def process_session_list(lines_list):
    session_list = set([])
    for line in lines_list:
        matches = re.findall(r"\[(.*?)]", line)
        if matches:
            try:
                datetime.strptime(matches[0], "%Y-%m-%d %H:%M:%S")
                session_list.add(matches[1])
            except ValueError:
                print(f"{matches}")
    return session_list


def filter_errors(lines_list):
    err_str = [line for line in lines_list if "ERROR" in line]
    lines_list = [line for line in lines_list if line not in err_str]
    return err_str + lines_list


def on_session_combo_changed(event, session_combo, log_filename):
    widget_manager.widget_lock("disable", "wait")
    selected_item = session_combo.get()
    print("Selected item is:", selected_item)
    lines_list = process_selected_item(selected_item, log_filename)
    lines_list = filter_errors(lines_list)
    show_result(lines_list)
    widget_manager.widget_lock("normal", "")


def on_combobox_changed(event, id_combo, log_filename):
    widget_manager.widget_lock("disable", "wait")
    widget_manager.widget_destroy(widget_manager.widgets_combo)
    selected_item = id_combo.get()
    print("Selected item is:", selected_item)
    lines_list = process_selected_item(selected_item, log_filename)
    session_label = add_widget("", tk.Label, text="Session List")
    session_list = process_session_list(lines_list)
    session_combo = add_widget("", ttk.Combobox, values=list(session_list))
    session_combo.bind("<<ComboboxSelected>>", on_session_combo_changed)
    show_result(lines_list)
    widget_manager.widget_lock("normal", "")


def initialize_widgets(log_filename, log_lines):
    create_search_section(log_filename, log_lines)
    create_id_list_section(log_lines)


def create_search_section(log_lines):
    """Create the search text section in the GUI."""
    message_label = widget_manager.add_widget(tk.Label, text=config.get('MESSAGE', 'search_text'))
    entry = widget_manager.add_widget(tk.Entry)
    search_button = widget_manager.add_widget(tk.Button, text="SearchAndCopy",
                                              command=lambda: copy_grep_results(log_lines, entry))


def create_id_list_section(log_filename, log_lines):
    """Create the ID list section in the GUI."""
    message_label = widget_manager.add_widget(tk.Label, text="ID list")
    id_list = make_id_list(log_lines)
    id_combo = widget_manager.add_widget(ttk.Combobox, values=list(id_list))
    id_combo.set(config.get('MESSAGE', 'combo_search'))
    bind_events_to_id_combo(id_combo, id_list, log_filename)


def bind_events_to_id_combo(id_combo, id_list, log_filename):
    id_combo.bind('<KeyRelease>', lambda event: search_ids(event, id_combo, id_list))
    id_combo.bind("<<ComboboxSelected>>",
                  lambda event: on_combobox_changed(event, id_combo, log_filename))

    widgets_combo = widget_manager.widgets_init_download.copy()
    widgets_combo.append(id_combo)


def search_ids(event, id_combo, id_list):
    search_text = id_combo.get()
    filtered_ids = [id for id in id_list if search_text.lower() in id.lower()]
    id_combo['values'] = filtered_ids


def show_result(lines_list):
    all_lines = ''.join(lines_list)
    result_filename = config.get('LOG_PATHS', 'result')
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(all_lines)
    webbrowser.open(result_filename)
    root.clipboard_clear()
    root.clipboard_append(all_lines)


def on_get_log_button_clicked():
    widget_manager.widget_destroy((combo, get_log_button))
    server_name = combo.get().upper()
    server_filelist = get_server_filelist(server_name)
    create_file_button(server_filelist, server_name)


def create_file_button(server_filelist, server):
    """Create a file download button with a server filelist combo box."""
    server_list_combo = widget_manager.add_widget(ttk.Combobox, values=server_filelist)
    download_button = widget_manager.add_widget(tk.Button, text=config.get("MESSAGE", 'download'),
                                                command=lambda: download_file(server_list_combo.get(), server))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Compassion Log v0.2")
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    combo = widget_manager.add_widget(ttk.Combobox, values=["dev", "prod1", "prod2"])
    get_log_button = widget_manager.add_widget(tk.Button, text="Get Log", command=on_get_log_button_clicked)
    root.mainloop()
