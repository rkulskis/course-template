#!/bin/bash
DIR=$1
PARENT_DIR=$(dirname "$DIR")
PUBLIC_REPO="../students-course-template"

if [ -f "$DIR/Makefile" ]; then
		make -C "$DIR"
fi 
if [ -d "$PUBLIC_REPO/$PARENT_DIR/$DIR" ]; then # get rid of old version
		rm -rf "$PUBLIC_REPO/$PARENT_DIR/$DIR"
fi

mkdir -p "$PUBLIC_REPO"/{pa,wa,lectures,labs,exams,quizzes}
for dir in pa wa lectures labs exams quizzes; do
	  cp "$dir/README.md" "$PUBLIC_REPO/$dir/";
done

echo "pa/problems
pa/*/*
!pa/*/starter_code
!pa/*/starter_code/**
autograder
solution
*.sol
*.out														
*.tex
*~
__pycache__
*.zip" > "$PUBLIC_REPO/.gitignore"

cp -rL "$DIR" "$PUBLIC_REPO/$PARENT_DIR"				# copy over new version
pushd "$PUBLIC_REPO"
git add .
git commit -m "publish"
git push;												# push to github
popd

