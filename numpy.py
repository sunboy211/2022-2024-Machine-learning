import numpy as np
import xlrd

read=xlrd.open_workbook('Z:\math\data.xlsx')
sheet1=read.sheet_by_index(0)
row=sheet1.nrows
col=sheet1.ncols
matrix=np.zeros((row,col))

for x in range(row):
    for y in range(col):
        matrix[x][y]=sheet1.cell_value(x,y)

A=np.mat([[0.049,0.01225,0.06125,0.01225,0.049,0.0245,0.03645,0.05,0.05,0.05,0.05,0.05,0.005,0.25,0.25]]).T
np.dot(matrix,A)