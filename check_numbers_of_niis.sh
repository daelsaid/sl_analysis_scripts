for x in `ls -d sl_story?_?pf_*/`; do echo `echo $x | cut -d/ -f1,3` `ls $x/*/*/*nii | wc -l`;done
