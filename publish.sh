#!/bin/bash
DIR=$1
PARENT_DIR=$(dirname "$DIR")
PUBLIC_REPO="../autograder-template-students"

if [ -f "$DIR/Makefile" ]; then
		make -C "$DIR"
fi 
if [ -d "$PUBLIC_REPO/$PARENT_DIR/$DIR" ]; then # get rid of old version
		rm -rf "$PUBLIC_REPO/$PARENT_DIR/$DIR"
fi
mkdir -p "$PUBLIC_REPO"/{pa,wa,lectures,labs,exams,quizzes}
cp -rL "$DIR" "$PUBLIC_REPO/$PARENT_DIR"				# copy over new version
pushd "$PUBLIC_REPO"
git add .
git commit -m "publish"
git push;												# push to github
popd

