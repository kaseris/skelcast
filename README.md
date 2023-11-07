# skelcast
Skeletal join forecasting

```
skelcast
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ ORIG_HEAD
│  ├─ branches
│  ├─ config
│  ├─ description
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  ├─ dataload
│  │     │  ├─ dev
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           ├─ HEAD
│  │           ├─ dev
│  │           └─ main
│  ├─ objects
│  │  ├─ 0b
│  │  │  └─ da740ea725a53759f72e2a1d4ec4cceb62cbe4
│  │  ├─ 0c
│  │  │  └─ 1cb655a4ed883aff9edf646fd37d525773b37f
│  │  ├─ 1f
│  │  │  └─ dd329c1b357486c6a4a4ad19131de721fbf20c
│  │  ├─ 28
│  │  │  └─ 207264e11acc233dee45c6b055f202da3b63fd
│  │  ├─ 57
│  │  │  └─ bd924d78b24bdfecf75b6e5344812c16cb35e5
│  │  ├─ 5c
│  │  │  └─ e18800358c7033142f6cef36aeec9b38bc3677
│  │  ├─ 68
│  │  │  └─ bc17f9ff2104a9d7b6777058bb4c343ca72609
│  │  ├─ 78
│  │  │  └─ a3ea6fd7f19b14568f7f2f322a1fd682477d23
│  │  ├─ 7f
│  │  │  └─ 40d6cb33b076fd0d5427900ec3108f7533b0e1
│  │  ├─ 99
│  │  │  └─ c4176c3439ebb8d5516a62cac75c999afb0997
│  │  ├─ 9d
│  │  │  └─ 1dcfdaf1a6857c5f83dc27019c7600e1ffaff8
│  │  ├─ a6
│  │  │  └─ 024cd831c2cae423d1d296f454bde001ad64e5
│  │  ├─ aa
│  │  │  └─ 609e071b5e61e04ccda97a16e424b20851c40a
│  │  ├─ c7
│  │  │  └─ b4df9f66cb58db3e9ccf6bfc4bf4a6f21e6ba9
│  │  ├─ ce
│  │  │  └─ 02cc4d0714bd17f15fee1ee5752f573ef0f526
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ f1
│  │  │  └─ 98518037b866424fc3e02949c3ef9bd44c304a
│  │  ├─ fe
│  │  │  ├─ 2564549a097f78685008f41b25d9e7d577b805
│  │  │  └─ 638f04858c96fad86b57a2743a88aacbe5f93b
│  │  ├─ ff
│  │  │  └─ 3616a965e5b425ae863703efc90ebba138f51b
│  │  ├─ info
│  │  └─ pack
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  ├─ dataload
│     │  ├─ dev
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ HEAD
│     │     ├─ dev
│     │     └─ main
│     └─ tags
├─ .gitattributes
├─ .gitignore
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ data
│  └─ missing
│     ├─ nturgb_d.txt
│     └─ nturgb_d_120.txt
├─ notebooks
│  └─ eda.ipynb
├─ poetry.lock
├─ pyproject.toml
├─ requirements.txt
└─ src
   └─ skelcast
      ├─ __init__.py
      ├─ data
      │  ├─ __init__.py
      │  └─ prepare_data.py
      ├─ models
      │  └─ __init__.py
      └─ utils
         ├─ __init__.py
         └─ data_processor.py

```
```
skelcast
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ ORIG_HEAD
│  ├─ branches
│  ├─ config
│  ├─ description
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  ├─ dataload
│  │     │  ├─ dev
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           ├─ HEAD
│  │           ├─ dev
│  │           └─ main
│  ├─ objects
│  │  ├─ 0b
│  │  │  └─ da740ea725a53759f72e2a1d4ec4cceb62cbe4
│  │  ├─ 0c
│  │  │  └─ 1cb655a4ed883aff9edf646fd37d525773b37f
│  │  ├─ 1f
│  │  │  └─ dd329c1b357486c6a4a4ad19131de721fbf20c
│  │  ├─ 28
│  │  │  └─ 207264e11acc233dee45c6b055f202da3b63fd
│  │  ├─ 57
│  │  │  └─ bd924d78b24bdfecf75b6e5344812c16cb35e5
│  │  ├─ 5c
│  │  │  └─ e18800358c7033142f6cef36aeec9b38bc3677
│  │  ├─ 68
│  │  │  └─ bc17f9ff2104a9d7b6777058bb4c343ca72609
│  │  ├─ 78
│  │  │  └─ a3ea6fd7f19b14568f7f2f322a1fd682477d23
│  │  ├─ 7f
│  │  │  └─ 40d6cb33b076fd0d5427900ec3108f7533b0e1
│  │  ├─ 99
│  │  │  └─ c4176c3439ebb8d5516a62cac75c999afb0997
│  │  ├─ 9d
│  │  │  └─ 1dcfdaf1a6857c5f83dc27019c7600e1ffaff8
│  │  ├─ a6
│  │  │  └─ 024cd831c2cae423d1d296f454bde001ad64e5
│  │  ├─ aa
│  │  │  └─ 609e071b5e61e04ccda97a16e424b20851c40a
│  │  ├─ c7
│  │  │  └─ b4df9f66cb58db3e9ccf6bfc4bf4a6f21e6ba9
│  │  ├─ ce
│  │  │  └─ 02cc4d0714bd17f15fee1ee5752f573ef0f526
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ f1
│  │  │  └─ 98518037b866424fc3e02949c3ef9bd44c304a
│  │  ├─ fe
│  │  │  ├─ 2564549a097f78685008f41b25d9e7d577b805
│  │  │  └─ 638f04858c96fad86b57a2743a88aacbe5f93b
│  │  ├─ ff
│  │  │  └─ 3616a965e5b425ae863703efc90ebba138f51b
│  │  ├─ info
│  │  └─ pack
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  ├─ dataload
│     │  ├─ dev
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ HEAD
│     │     ├─ dev
│     │     └─ main
│     └─ tags
├─ .gitattributes
├─ .gitignore
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ data
│  └─ missing
│     ├─ nturgb_d.txt
│     └─ nturgb_d_120.txt
├─ notebooks
│  └─ eda.ipynb
├─ poetry.lock
├─ pyproject.toml
├─ requirements.txt
└─ src
   └─ skelcast
      ├─ __init__.py
      ├─ data
      │  ├─ __init__.py
      │  └─ prepare_data.py
      ├─ models
      │  └─ __init__.py
      └─ utils
         ├─ __init__.py
         └─ data_processor.py

```