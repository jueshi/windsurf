import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt
import subprocess

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Run main.py with File/Directory Input')
        self.resize(600, 350)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input mode (File/Dir or URL)
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        mode_layout = QHBoxLayout()
        self.file_radio = QRadioButton('文件/目录')
        self.file_radio.setChecked(True)
        self.url_radio = QRadioButton('URL')
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.file_radio)
        self.mode_group.addButton(self.url_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addWidget(self.url_radio)
        layout.addLayout(mode_layout)
        self.file_radio.toggled.connect(self.toggle_input_mode)

        # Input file/dir
        input_layout = QHBoxLayout()
        self.input_label = QLabel('输入文件/目录:')
        self.input_edit = QLineEdit()
        self.input_btn = QPushButton('选择...')
        self.input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_btn)
        layout.addLayout(input_layout)

        # URL input
        self.url_layout = QHBoxLayout()
        self.url_label = QLabel('输入URL:')
        self.url_edit = QLineEdit()
        self.url_layout.addWidget(self.url_label)
        self.url_layout.addWidget(self.url_edit)
        layout.addLayout(self.url_layout)
        self.url_label.hide()
        self.url_edit.hide()
        self.url_layout.setEnabled(False)

        # Language
        lang_layout = QHBoxLayout()
        self.lang_label = QLabel('语言:')
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['Chinese', 'English'])
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)

        # Output dir
        output_layout = QHBoxLayout()
        self.output_label = QLabel('输出目录:')
        self.output_edit = QLineEdit()
        self.output_edit.setText('output')  # Set default output folder
        self.output_btn = QPushButton('选择...')
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        # Run button
        self.run_btn = QPushButton('运行 main.py')
        self.run_btn.clicked.connect(self.run_main)
        layout.addWidget(self.run_btn)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        self.setLayout(layout)

    def toggle_input_mode(self):
        if self.file_radio.isChecked():
            self.input_label.show()
            self.input_edit.show()
            self.input_btn.show()
            self.url_label.hide()
            self.url_edit.hide()
            self.url_layout.setEnabled(False)
        else:
            self.input_label.hide()
            self.input_edit.hide()
            self.input_btn.hide()
            self.url_label.show()
            self.url_edit.show()
            self.url_layout.setEnabled(True)

    def select_input(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', '', 'All Files (*)', options=options)
        if not file_path:
            dir_path = QFileDialog.getExistingDirectory(self, '选择目录', '', options=options)
            if dir_path:
                self.input_edit.setText(dir_path)
        else:
            self.input_edit.setText(file_path)

    def select_output(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, '选择输出目录', 'output', options=options)
        if dir_path:
            self.output_edit.setText(dir_path)

    def run_main(self):
        language = self.lang_combo.currentText()
        output_dir = self.output_edit.text().strip()
        main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
        if not os.path.exists(main_py):
            self.output_area.append('找不到 main.py 文件!')
            return
        if self.file_radio.isChecked():
            input_path = self.input_edit.text().strip()
            if not input_path or not output_dir:
                self.output_area.append('请填写输入文件/目录和输出目录!')
                return
            cmd = [sys.executable, main_py, '--file', input_path, '--language', language, '-o', output_dir]
        else:
            url = self.url_edit.text().strip()
            if not url or not output_dir:
                self.output_area.append('请填写URL和输出目录!')
                return
            cmd = [sys.executable, main_py, '--url', url, '--language', language, '-o', output_dir]
        self.output_area.append(f'运行命令: {" ".join(cmd)}')
        self.output_area.append('...请稍候...\n')
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(main_py)  # Ensure cwd is main.py's folder
            )
            stdout, stderr = proc.communicate()
            if stdout:
                self.output_area.append(stdout)
            if stderr:
                self.output_area.append('<span style="color:red">'+stderr+'</span>')
        except Exception as e:
            self.output_area.append(f'运行出错: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
