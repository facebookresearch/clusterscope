# Release

1. Checkout and pull latest main
```
$ git checkout main
$ git pull origin main
```
2. Tag
```
$ git tag -a v0.0.0 -m "Release 0.0.0"
```
3. Push the tag to github
```
$ git push origin v0.0.0
```
4. [Find and run the action related to publishing the branch to PyPI](https://github.com/facebookresearch/clusterscope/actions). This requires maintainer approval.
