with import <nixpkgs> {};

(python37.buildEnv.override {
  extraLibs = with pkgs.python37Packages;
    [
      pytorch
      autograd
      numpy
      scipy
      matplotlib
      pillow

      # Utilities
      yapf
      python-language-server
    ];

  ignoreCollisions = true;
}).env
