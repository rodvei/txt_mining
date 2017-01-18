import win32com.client
import docx2txt

# file_path = 'D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\AdvokatfirmaBrækhusDege\\FraFirmaet\\Arbeidsrettsaker\\'
# filename = '9005 - Ansettelsesavtale.doc'
# filename = '8998 - Sluttavtale - revidert-1.docx'

def document_to_text(filename, file_path):
	if file_path[-1:]!='\\' and filename[:1]!='\\':
		file_path+='\\'
	# path prep
	if filename[-4:] == ".doc":
		app = win32com.client.Dispatch('Word.Application')
		app.Visible = False
		doc = app.Documents.Open(file_path+filename)
		text = doc.Content.Text
		doc.Close()
		app.Quit()
		return(text)
	elif filename[-5:] == ".docx":
		text = docx2txt.process(file_path+filename)
		return(text)
	elif filename[-4:] == ".pdf":
		# future improvements
		return('')
		