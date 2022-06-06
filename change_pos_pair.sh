for f in */*.pkl; do
  fnew="${f%%.pkl}_sec_5_deg_10.pkl";
  mv $f $fnew
done

# for file in *_h.png ; do mv "$file" "${file%%_h.png}_half.png" ; done
