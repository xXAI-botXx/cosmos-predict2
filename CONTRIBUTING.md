# How to Contribute

We'd love to receive your patches and contributions. Please keep your PRs as draft until such time that you would like us to review them.

## Testing

Install system dependencies:

[just](https://just.systems/man/en/pre-built-binaries.html#pre-built-binaries)

```shell
mkdir -p "$HOME/.local"
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
```

To test your changes locally, run

```shell
just test
```

This will also run auto-fixes and linting. We recommend that you commit your changes first.

To see all available commands, run

```shell
just
```

## Releasing

To release, you must be a maintainer of the [cosmos-predict2 package](https://pypi.org/project/cosmos-predict2/).

Run pre-release check:

```shell
just release-check
```

Commit any changes.

Release to PyPI (omit `<pypi_token>` to dry-run):

```shell
just release <pypi_token>
```

Push the new tag to GitHub:

```shell
git push git@github.com:nvidia-cosmos/cosmos-predict2.git "v$(uv version --short)"
```

Merge the new commit to GitHub.

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
