from FolderCleaner import FolderCleaner

name = "Vilhjalmur R. Vilhjalmsson"

cleaner = FolderCleaner(name)

cleaner.fetchAll()
cleaner.writeToJson()
