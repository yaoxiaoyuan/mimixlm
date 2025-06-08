import sys
import tkinter as tk
from tkinter import messagebox, font
import torch
from mimixlm import load_model_with_processor,build_generation_inputs

class SudokuGUI:
    def __init__(self, root, model_path):

        self.model, self.tokenizer = load_model_with_processor(model_path)[:2]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        self.root = root
        self.root.title("Sudoku Pro")
        self.root.geometry("550x650")
        self.cell_size = 50
        self.selected = (-1, -1)
        self.current_puzzle = [[0]*9 for _ in range(9)]  
        
        self.colors = {
            "bg": "#FFFFFF",
            "cell_bg": "#FAFAFA",
            "fixed": "#757575",
            "input": "#2196F3",
            "highlight": "#BBDEFB",
            "line": "#E0E0E0",
            "bold_line": "#616161"
        }
        
        self.init_ui()
        self.reset_puzzle()
        self.setup_keyboard_bindings()


    def init_ui(self):
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 工具栏
        toolbar = tk.Frame(main_frame, bg=self.colors["bg"])
        toolbar.pack(pady=(0, 15))
        
        self.create_button(toolbar, "Generate puzzle", "#4CAF50", self.generate_puzzle).grid(row=0, column=0, padx=5)
        self.create_button(toolbar, "Solve puzzle", "#2196F3", self.solve).grid(row=0, column=1, padx=5)
        self.create_button(toolbar, "Clear all", "#FF9800", self.new_game).grid(row=0, column=2, padx=5)

        # 数独网格
        self.grid_canvas = tk.Canvas(main_frame, 
                                   width=self.cell_size*9,
                                   height=self.cell_size*9,
                                   bg=self.colors["bg"],
                                   highlightthickness=0)
        self.grid_canvas.pack()
        self.draw_grid()
        self.create_cells()

    def generate_puzzle(self):
        self.new_game()
        try_cnt = 0
        while True:
            inputs = [""]
            x = build_generation_inputs(self.model, inputs, self.tokenizer, self.device)[0]
            with torch.no_grad():
                gen_text = ""
                for search_states in self.model.search(x, logits_processors=[]):
                    search_states["text_buffer"] = []
                    for ids in search_states["hypothesis"].cpu().numpy():
                        search_states["text_buffer"].append(self.tokenizer.decode(ids))
                    gen_text = search_states["text_buffer"][0]
                    
                    words = gen_text.split()
                    
                    if len(words) <= 81:
                        i = (len(words) - 1) // 9
                        j = (len(words) - 1) % 9
                        num = int(words[-1])
                        if num != 0:
                            self.grid_canvas.itemconfig(self.cells[i][j], text=str(num))
                        self.puzzle[i][j] = num

                    else:
                        i = (len(words) - 82) // 9
                        j = (len(words) - 82) % 9
                        num = int(words[-1])
                        self.current_puzzle[i][j] = num
                        if self.puzzle[i][j] == 0:
                            self.grid_canvas.itemconfig(
                            self.cells[i][j],
                            text=str(self.current_puzzle[i][j]),
                            fill=self.colors["input"]
                            )
                    self.update_highlight()
                    self.root.update() 
                    
            try_cnt += 1
            print(f"try generate {try_cnt} times, {gen_text}")
            if self.solve(max_try_cnt=3):
                print("Valid success!")
                self.current_puzzle = [row[:] for row in self.puzzle]
                for i in range(9):
                    for j in range(9):
                        if self.puzzle[i][j] == 0:
                            self.grid_canvas.itemconfig(self.cells[i][j], text="")
                self.update_highlight()
                self.root.update()
                messagebox.showinfo("Check", "Success!")
                break
            else:
                print("Not valid!")
                self.new_game()

    def setup_keyboard_bindings(self):
        # 绑定键盘事件
        self.root.bind("<Key>", self.handle_key_press)
        self.root.bind("<BackSpace>", self.handle_delete)
        self.root.bind("<Delete>", self.handle_delete)
        

    def draw_grid(self):
        """绘制完整的网格系统"""
        # 绘制粗线分隔3x3宫格
        for i in range(0, 10):
            line_width = 3 if i % 3 == 0 else 1
            color = self.colors["bold_line"] if i % 3 == 0 else self.colors["line"]
            
            # 水平线
            self.grid_canvas.create_line(
                0, i*self.cell_size,
                self.cell_size*9, i*self.cell_size,
                width=line_width, fill=color
            )
            
            # 垂直线
            self.grid_canvas.create_line(
                i*self.cell_size, 0,
                i*self.cell_size, self.cell_size*9,
                width=line_width, fill=color
            )

    def create_cells(self):
        """创建可交互的单元格"""
        self.cells = []
        self.cell_rects = []
        bold_font = font.Font(family="Arial", size=18, weight="bold")
        
        for i in range(9):
            row = []
            rect_row = []
            for j in range(9):
                # 创建半透明背景矩形（不覆盖网格线）
                x1 = j * self.cell_size + 1
                y1 = i * self.cell_size + 1
                x2 = x1 + self.cell_size - 2
                y2 = y1 + self.cell_size - 2
                
                rect = self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=self.colors["cell_bg"],
                    outline="",
                    tags=("cell_bg", f"cell_{i}_{j}")
                )
                rect_row.append(rect)
                
                # 创建数字文本
                text = self.grid_canvas.create_text(
                    x1 + (self.cell_size-2)//2,
                    y1 + (self.cell_size-2)//2,
                    text="",
                    font=bold_font,
                    tags=("cell", f"cell_{i}_{j}")
                )
                row.append(text)
                
                # 绑定点击事件
                self.grid_canvas.tag_bind(rect, "<Button-1>", lambda e, i=i, j=j: self.select_cell(i, j))
                self.grid_canvas.tag_bind(text, "<Button-1>", lambda e, i=i, j=j: self.select_cell(i, j))
                
            self.cells.append(row)
            self.cell_rects.append(rect_row)

    def select_cell(self, i, j):
        """选择单元格"""
        self.selected = (i, j)
        self.update_highlight()

    def update_highlight(self):
        """更新高亮显示"""                
        # 更新单元格颜色
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j] != 0:
                    self.grid_canvas.itemconfig(
                        self.cell_rects[i][j],
                        fill=self.colors["fixed"],
                        tags="fixed"
                    )
                elif self.selected == (i,j):
                    self.grid_canvas.itemconfig(
                        self.cell_rects[i][j],
                        fill=self.colors["highlight"],
                        tags="highlight"
                    )
                else:
                    self.grid_canvas.itemconfig(
                        self.cell_rects[i][j],
                        fill=self.colors["cell_bg"],
                        tags="cell_bg"
                    )

    def handle_key_press(self, event):
        """处理键盘输入"""
        if self.selected == (-1, -1):
            return
        
        char = event.char
        if char >= "1" and char <= "9":
            self.input_number(int(char))
        elif char == "0" or event.keysym in ["BackSpace", "Delete"]:
            self.delete_number()

    def handle_delete(self, event=None):
        self.delete_number()

    def input_number(self, num):
        """输入数字"""
        i, j = self.selected
        self.puzzle[i][j] = num
        self.grid_canvas.itemconfig(self.cells[i][j], text=str(num))
        self.update_highlight()

    def delete_number(self):
        """删除数字"""
        i, j = self.selected
        self.puzzle[i][j] = 0
        self.grid_canvas.itemconfig(self.cells[i][j], text="")
        self.update_highlight()

    def reset_puzzle(self):
        """加载初始谜题"""
        self.puzzle = [[0]*9 for _ in range(9)]
        
        self.current_puzzle = [row[:] for row in self.puzzle]
        
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j] != 0:
                    self.grid_canvas.itemconfig(
                        self.cells[i][j],
                        text=str(self.puzzle[i][j]),
                        fill=self.colors["fixed"]
                    )
                else:
                    self.grid_canvas.itemconfig(
                        self.cells[i][j],
                        text="",
                        fill=self.colors["cell_bg"]
                    )
        
        self.update_highlight()

    def create_button(self, parent, text, color, command):
        return tk.Button(parent,
                        text=text,
                        command=command,
                        bg=color,
                        fg="white",
                        font=("Arial", 12, "bold"),
                        padx=15,
                        pady=5,
                        bd=0,
                        relief="flat",
                        activebackground=color)

    def check_solution(self):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        blocks = [set() for _ in range(9)]
        flag = True
        for i in range(81):
                                                                            
            row = i // 9                                                    
            col = i % 9                                                     
            block = (row // 3) * 3 + (col // 3)                             
                        
            num = self.current_puzzle[row][col]
            if num in rows[row] or num in cols[col] or num in blocks[block]:
                #print(row, col, num)
                flag = False                                                
                                                                            
            rows[row].add(num)                                              
            cols[col].add(num)                                              
            blocks[block].add(num)                                          
                                                                            
        return flag        

    def solve(self, max_try_cnt=20):
        self.clear_solve()
        self.root.update()
        
        q = " ".join([str(x) for li in self.puzzle for x in li])
        print(q)
        
        inputs = [q]
        x = build_generation_inputs(self.model, inputs, self.tokenizer, self.device)[0]
        tokens = [] 
        try_cnt = 0
        while try_cnt < max_try_cnt:
            with torch.no_grad():
                gen_text = ""
                for search_states in self.model.search(x, logits_processors=[]):
                    search_states["text_buffer"] = []
                    for ids in search_states["hypothesis"].cpu().numpy():
                        search_states["text_buffer"].append(self.tokenizer.decode(ids))
                    gen_text = search_states["text_buffer"][0]
                    
                    words = gen_text.split()
                    i = (len(words) - 1) // 9
                    j = (len(words) - 1) % 9
                    num = int(words[-1])
                    
                    if self.puzzle[i][j] == 0:
                        self.current_puzzle[i][j] = num
                        self.grid_canvas.itemconfig(
                            self.cells[i][j],
                            text=str(self.current_puzzle[i][j]),
                            fill=self.colors["input"]
                        )
                        self.update_highlight()
                        self.root.update()
                        
            try_cnt += 1
            print(f"try solve {try_cnt} times, {gen_text}")
            if self.check_solution():
                print("Solve Success!")
                messagebox.showinfo("Check", "Solve Success!")
                return True
            else:
                print("Solve Failed!")
                self.clear_solve()
        return False    

    def clear_solve(self):
        self.current_puzzle = [row[:] for row in self.puzzle]
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j] != 0:
                    self.grid_canvas.itemconfig(
                        self.cells[i][j],
                        text=str(self.puzzle[i][j]),
                    )
                else:
                    self.grid_canvas.itemconfig(
                        self.cells[i][j],
                        text="",
                    )
        self.update_highlight()

    def new_game(self):
        # 重置游戏状态
        self.selected = (-1, -1)
        self.grid_canvas.delete("highlight")
        self.reset_puzzle()
        self.root.update()

if __name__ == "__main__":
    model_path = sys.argv[1]
    root = tk.Tk()
    app = SudokuGUI(root, model_path)
    root.mainloop()