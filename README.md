

```
conda activate zwq-torch-gpu
python.exe main.py  --input_dir E:\ChromeDownload\sudoku--1102  --output_dir  E:\ChromeDownload\sudoku--1102-1104
# 1/8, OK


python.exe main.py  --input_dir  E:\ChromeDownload\1002--crop --output_dir  E:\ChromeDownload\1002--crop-1104

```


运行的时候，有一些没有检测到图片，可能是utils.py 代码的检测的鲁棒性不够。而且在处理第7张图片的时候，卡死了。

请基于以上的信息，生成更加鲁棒的代码

```
[2/103] Processing '06785fdcabf11fae.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[3/103] Processing '07bb01c8d955513c.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[4/103] Processing '08b8baef75c1921d.jpg'...
  [ERROR] Unexpected error: No grid contour candidates were found in image
[5/103] Processing '0a8375eee99cf647.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[6/103] Processing '0ab69e46f6d35df7.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[7/103] Processing '0cf5ea0b7e3a8d4d.jpg'...
Traceback (most recent call last):
  File "C:\HF-Models\sudoku_solver\main.py", line 125, in <module>
    main()
  File "C:\HF-Models\sudoku_solver\main.py", line 106, in main
    annotated_img = solve_sudoku_puzzle(loaded_model, img_fpath)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\main.py", line 45, in solve_sudoku_puzzle
    solver.solve()
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  [Previous line repeated 40 more times]
  File "C:\HF-Models\sudoku_solver\utils.py", line 107, in solve
    next_empty_pos = self.find_next_empty()
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 57, in find_next_empty
    if self.board[row][col] == empty_val:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

