import os
import javabridge as jv
import bioformats

path = r'C:\Users\M1SRH\Documents\Software\BioFormats_Package\5.1.4\bioformats_package.jar'
#path = r'c:\Python27\Lib\site-packages\bioformats\jars\loci_tools.jar'
jars = jv.JARS + [path]
jv.start_vm(class_path=jars, max_heap_size='2G')
paths = jv.JClassWrapper('java.lang.System').getProperty('java.class.path').split(";")

for path in paths:
    print "%s: %s" %("exists" if os.path.isfile(path) else "missing", path)

filename = r'c:\Users\M1SRH\Documents\Z-Stack.czi'
rdr = bioformats.get_image_reader(None, path=filename)