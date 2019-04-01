from FolderCleaner import FolderCleaner

name = "Vilhjalmur R. Vilhjalmsson"

cleaner = FolderCleaner(name, full=False)

cleaner.fetchAll()
cleaner.writeToJson()
