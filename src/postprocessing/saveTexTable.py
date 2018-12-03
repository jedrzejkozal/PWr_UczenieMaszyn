from postprocessing.utils import getFirstItemFromDict

class SaveTexTable:

    def saveTable(self, results, selector, fileName):
        rowsList = [[" "]]

        for name, _ in getFirstItemFromDict(results).items():
            rowsList[0].append(name)

        for extractionName, extractionResults in results.items():
            rowsList.append([extractionName])
            for classifierName, scores in extractionResults.items():
                floatToSave = round(selector(scores), 2)
                rowsList[-1].append('{0:.2f}'.format(floatToSave))

        self.saveTexTable(rowsList, fileName)


    def saveTexTable(self, rowsList, fileName):
        path = "../doc/tables/"+fileName+".tex"
        f = open(path, 'w')
        self.putTableToFile(rowsList, f)
        f.close()


    def putTableToFile(self, rowsList, file):
        tableHeader = "\\begin{tabular}{|r|" + "l|"*(len(rowsList[0])-1) + "}\n"
        file.write(tableHeader)
        file.write("  \\hline\n")

        for row in rowsList:
            self.saveSingleRow(row, file)

        file.write("\end{tabular}\n")


    def saveSingleRow(self, row, file):
        rowString = self.getRowString(row)
        stringToSave = rowString[:-2] + "\\\\\n"
        file.write(stringToSave)
        file.write("  \\hline\n")


    def getRowString(self, row):
        rowString = "  "
        for elem in row:
            rowString = rowString + elem + " & "

        return rowString
