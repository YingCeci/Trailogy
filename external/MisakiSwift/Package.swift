// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MisakiSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "MisakiSwift",
      type: .dynamic,
      targets: ["MisakiSwift"]
    ),
  ],
  dependencies: [
    // VENDORED CHANGE: relaxed MLX pin so we can coexist with mlx-swift-lm 3.x
    .package(url: "https://github.com/ml-explore/mlx-swift", "0.30.2" ..< "1.0.0"),
    // Sibling vendored package, referenced by relative path so the whole
    // dep graph uses one MLXUtilsLibrary instance.
    .package(path: "../MLXUtilsLibrary")
  ],
  targets: [
    .target(
      name: "MisakiSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
     ],
     resources: [
      .copy("../../Resources/")
     ]
    ),
    .testTarget(
      name: "MisakiSwiftTests",
      dependencies: ["MisakiSwift"]
    ),
  ]
)
