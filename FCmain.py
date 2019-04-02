from FolderCleaner import FolderCleaner

name = "Vilhjalmur R. Vilhjalmsson"

cleaner = FolderCleaner(name, full=True)

cleaner.fetchAll()
cleaner.writeToJson()
