main_crate := 'mlt-core'
just := quote(just_executable())

# if running in CI, treat warnings as errors by setting RUSTFLAGS and RUSTDOCFLAGS to '-D warnings' unless they are already set
# Use `CI=true just ci-test` to run the same tests as in GitHub CI.
# Use `just ci-env-info` to see the current values of RUSTFLAGS and RUSTDOCFLAGS
ci_mode := if env('CI', '') != '' {'1'} else {''}
export RUSTFLAGS := env('RUSTFLAGS', if ci_mode == '1' {'-D warnings'} else {''})
export RUSTDOCFLAGS := env('RUSTDOCFLAGS', if ci_mode == '1' {'-D warnings'} else {''})
export RUST_BACKTRACE := env('RUST_BACKTRACE', if ci_mode == '1' {'1'} else {'0'})

_default: (just '--list' 'rust')

[private]
just *args:
    {{just}} {{args}}

# Run integration tests and save its output as the new expected output
bless *args: (just 'cargo-install' 'cargo-insta')
    cargo insta test --accept --unreferenced=delete --all-features {{args}}

# Run benchmarks
bench:
    cargo bench --all-targets --all-features

alias b := build
# Build the project
build:
    cargo build --all-targets --all-features

alias c := check
# Quick compile without building a binary
check: (just 'cargo-install' 'cargo-hack')
    cargo hack check --all-targets --each-feature --workspace
    cd mlt-core/fuzz && cargo check --all-features

# Generate code coverage report to upload to codecov.io
ci-coverage: ci-env-info (just 'java::gen-snippets')
    mkdir -p target/llvm-cov
    # ATTENTION: the output-path here is used in the CI workflow
    {{just}} rust::coverage '--codecov --output-path target/llvm-cov/codecov.info'

# Run minimal subset of tests to ensure compatibility with MSRV
ci-check: ci-env-info check

# Print environment info for CI
ci-env-info:
    @echo "::group::General environment Information"
    @echo "Running for '{{main_crate}}' crate {{if ci_mode == '1' {'in CI mode'} else {'in dev mode'} }} on {{os()}} / {{arch()}}"
    @echo "PWD {{justfile_directory()}}"
    {{just}} --version
    rustc --version
    cargo --version
    rustup --version
    @echo "RUSTFLAGS='$RUSTFLAGS'"
    @echo "RUSTDOCFLAGS='$RUSTDOCFLAGS'"
    @echo "RUST_BACKTRACE='$RUST_BACKTRACE'"
    @echo "::endgroup::"

    @echo "::group::C++ compiler info"
    g++ --version || echo "g++ not found"
    dpkg -l | grep -E '^ii\s+(gcc|g\+\+)' || echo "dpkg not available"
    gcc -march=native -Q --help=target 2>&1 | grep -E '^\s+-(march|mtune|mvaes|mvpclmulqdq|msha|mavx|mbmi|mfma)' || echo "Failed to get native march flags"
    @echo "::endgroup::"

    @echo "::group::lscpu"
    lscpu
    @echo "::endgroup::"

    @echo "::group::/proc/cpuinfo"
    cat /proc/cpuinfo
    @echo "::endgroup::"

# Debug AMD AVX SIGILL issue - runs only the failing test with maximum diagnostics
# See: https://github.com/maplibre/maplibre-tile-spec/issues/1016
ci-debug-amd-avx:
    #!/usr/bin/env bash
    set -euo pipefail

    DEBUG_DIR="target/amd-avx-debug"
    mkdir -p "$DEBUG_DIR"

    echo "============================================================"
    echo "DEBUG: AMD AVX SIGILL Investigation"
    echo "Issue: https://github.com/maplibre/maplibre-tile-spec/issues/1016"
    echo "============================================================"

    # Collect all environment info to a file
    ENV_FILE="$DEBUG_DIR/environment.txt"
    echo "=== Environment Info ===" > "$ENV_FILE"
    echo "Date: $(date -u)" >> "$ENV_FILE"
    echo "Hostname: $(hostname)" >> "$ENV_FILE"
    echo "Kernel: $(uname -a)" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== Rust Toolchain ===" >> "$ENV_FILE"
    rustc --version >> "$ENV_FILE"
    cargo --version >> "$ENV_FILE"
    echo "RUSTFLAGS=$RUSTFLAGS" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== C++ Compiler ===" >> "$ENV_FILE"
    which g++ >> "$ENV_FILE" 2>&1 || echo "g++ not found" >> "$ENV_FILE"
    g++ --version >> "$ENV_FILE" 2>&1 || echo "g++ version failed" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"
    which clang++ >> "$ENV_FILE" 2>&1 || echo "clang++ not found" >> "$ENV_FILE"
    clang++ --version >> "$ENV_FILE" 2>&1 || echo "clang++ version failed" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== Installed GCC/G++ packages ===" >> "$ENV_FILE"
    dpkg -l | grep -E '^ii\s+(gcc|g\+\+)' >> "$ENV_FILE" 2>&1 || echo "dpkg not available" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== Active compiler symlinks ===" >> "$ENV_FILE"
    ls -la $(which gcc 2>/dev/null) $(which g++ 2>/dev/null) >> "$ENV_FILE" 2>&1 || true
    echo "" >> "$ENV_FILE"

    echo "=== libc version ===" >> "$ENV_FILE"
    ldd --version >> "$ENV_FILE" 2>&1 || echo "ldd not found" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== CMake ===" >> "$ENV_FILE"
    cmake --version >> "$ENV_FILE" 2>&1 || echo "cmake not found" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== GCC -march=native expansion ===" >> "$ENV_FILE"
    gcc -march=native -Q --help=target 2>&1 | grep -E '^\s+-(march|mtune|msse|mavx|mfma|mbmi|mvaes|mvpclmulqdq|msha)' >> "$ENV_FILE" || echo "Failed to get gcc flags" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== Environment variables affecting build ===" >> "$ENV_FILE"
    echo "CC=${CC:-<unset>}" >> "$ENV_FILE"
    echo "CXX=${CXX:-<unset>}" >> "$ENV_FILE"
    echo "CFLAGS=${CFLAGS:-<unset>}" >> "$ENV_FILE"
    echo "CXXFLAGS=${CXXFLAGS:-<unset>}" >> "$ENV_FILE"
    echo "CMAKE_ARGS=${CMAKE_ARGS:-<unset>}" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== Virtualization ===" >> "$ENV_FILE"
    cat /sys/devices/virtual/dmi/id/sys_vendor >> "$ENV_FILE" 2>&1 || echo "sys_vendor: N/A" >> "$ENV_FILE"
    systemd-detect-virt >> "$ENV_FILE" 2>&1 || echo "systemd-detect-virt: N/A" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    echo "=== OS Release ===" >> "$ENV_FILE"
    cat /etc/os-release >> "$ENV_FILE" 2>&1 || echo "os-release: N/A" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"

    # Save CPU info separately
    echo "=== lscpu ===" > "$DEBUG_DIR/lscpu.txt"
    lscpu >> "$DEBUG_DIR/lscpu.txt" 2>&1

    echo "=== /proc/cpuinfo ===" > "$DEBUG_DIR/cpuinfo.txt"
    cat /proc/cpuinfo >> "$DEBUG_DIR/cpuinfo.txt" 2>&1

    # Extract just the flags for easy comparison
    echo "=== CPU Flags ===" > "$DEBUG_DIR/cpu_flags.txt"
    grep -m1 'flags' /proc/cpuinfo | tr ' ' '\n' | sort >> "$DEBUG_DIR/cpu_flags.txt" 2>&1

    # AVX-specific flags
    echo "=== AVX-related flags ===" > "$DEBUG_DIR/avx_flags.txt"
    grep -m1 'flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse|avx|fma|bmi)' | sort >> "$DEBUG_DIR/avx_flags.txt" 2>&1

    # CPU microcode
    echo "=== CPU Microcode ===" > "$DEBUG_DIR/microcode.txt"
    cat /sys/devices/system/cpu/cpu0/microcode/version >> "$DEBUG_DIR/microcode.txt" 2>&1 || echo "N/A" >> "$DEBUG_DIR/microcode.txt"

    # CPU vulnerabilities (mitigations can affect behavior)
    echo "=== CPU Vulnerabilities ===" > "$DEBUG_DIR/vulnerabilities.txt"
    for f in /sys/devices/system/cpu/vulnerabilities/*; do
        echo "$(basename $f): $(cat $f)" >> "$DEBUG_DIR/vulnerabilities.txt" 2>&1
    done

    # Test basic AVX2 execution
    echo "=== Testing basic AVX2 execution ===" | tee "$DEBUG_DIR/avx_test.txt"
    cat > /tmp/avx_test.cpp << 'CPPEOF'
    #include <immintrin.h>
    #include <cstdio>
    int main() {
        printf("Testing AVX2 instructions...\n");
        __m256i a = _mm256_set1_epi32(42);
        __m256i b = _mm256_add_epi32(a, a);
        int result = _mm256_extract_epi32(b, 0);
        printf("AVX2 _mm256_add_epi32 result: %d (expected 84)\n", result);
        return (result == 84) ? 0 : 1;
    }
    CPPEOF

    g++ -march=native -o /tmp/avx_test /tmp/avx_test.cpp 2>&1 | tee -a "$DEBUG_DIR/avx_test.txt"
    /tmp/avx_test 2>&1 | tee -a "$DEBUG_DIR/avx_test.txt" || echo "AVX2 test FAILED with exit code $?" | tee -a "$DEBUG_DIR/avx_test.txt"

    # Test VAES (Vector AES-NI with YMM 256-bit registers) - likely culprit on AMD Zen3 under Hyper-V
    echo "=== Testing VAES (256-bit vector AES) execution ===" | tee "$DEBUG_DIR/vaes_test.txt"
    cat > /tmp/vaes_test.cpp << 'CPPEOF'
    #include <immintrin.h>
    #include <cstdio>
    int main() {
        printf("Testing VAES (256-bit vector AES-NI) instructions...\n");
        __m256i a = _mm256_setzero_si256();
        __m256i k = _mm256_setzero_si256();
        __m256i r = _mm256_aesenc_epi128(a, k);
        (void)r;
        printf("VAES _mm256_aesenc_epi128: OK\n");
        return 0;
    }
    CPPEOF
    if g++ -march=native -mvaes -o /tmp/vaes_test /tmp/vaes_test.cpp 2>&1 | tee -a "$DEBUG_DIR/vaes_test.txt"; then
        /tmp/vaes_test 2>&1 | tee -a "$DEBUG_DIR/vaes_test.txt" || echo "VAES test FAILED with exit code $? (SIGILL if illegal instruction)" | tee -a "$DEBUG_DIR/vaes_test.txt"
    else
        echo "VAES test: compiler does not support -mvaes" | tee -a "$DEBUG_DIR/vaes_test.txt"
    fi

    # Test VPCLMULQDQ (Vector Carryless Multiply with YMM 256-bit registers) - another likely culprit
    echo "=== Testing VPCLMULQDQ (256-bit vector carryless multiply) execution ===" | tee "$DEBUG_DIR/vpclmulqdq_test.txt"
    cat > /tmp/vpclmulqdq_test.cpp << 'CPPEOF'
    #include <immintrin.h>
    #include <cstdio>
    int main() {
        printf("Testing VPCLMULQDQ (256-bit vector carryless multiply) instructions...\n");
        __m256i a = _mm256_setzero_si256();
        __m256i b = _mm256_setzero_si256();
        __m256i r = _mm256_clmulepi64_epi128(a, b, 0);
        (void)r;
        printf("VPCLMULQDQ _mm256_clmulepi64_epi128: OK\n");
        return 0;
    }
    CPPEOF
    if g++ -march=native -mvpclmulqdq -o /tmp/vpclmulqdq_test /tmp/vpclmulqdq_test.cpp 2>&1 | tee -a "$DEBUG_DIR/vpclmulqdq_test.txt"; then
        /tmp/vpclmulqdq_test 2>&1 | tee -a "$DEBUG_DIR/vpclmulqdq_test.txt" || echo "VPCLMULQDQ test FAILED with exit code $? (SIGILL if illegal instruction)" | tee -a "$DEBUG_DIR/vpclmulqdq_test.txt"
    else
        echo "VPCLMULQDQ test: compiler does not support -mvpclmulqdq" | tee -a "$DEBUG_DIR/vpclmulqdq_test.txt"
    fi

    # Show what instructions gcc generates with -march=native
    echo "=== GCC -march=native code generation ===" > "$DEBUG_DIR/gcc_codegen.txt"
    cat > /tmp/codegen_test.cpp << 'CPPEOF'
    #include <immintrin.h>
    extern "C" __m256i test_avx2(__m256i a) {
        return _mm256_add_epi32(a, a);
    }
    CPPEOF

    g++ -march=native -S -o /tmp/codegen_test.s /tmp/codegen_test.cpp 2>&1
    cat /tmp/codegen_test.s >> "$DEBUG_DIR/gcc_codegen.txt" 2>&1

    echo ""
    echo "=== Building test binary ==="
    # Build only the mlt-core tests
    cargo test -p mlt-core --no-run --all-features 2>&1 | tee "$DEBUG_DIR/build_output.txt"

    # Find the test binary
    TEST_BIN=$(cargo test -p mlt-core --no-run --all-features --message-format=json 2>/dev/null | \
        jq -r 'select(.executable != null and .target.name == "mlt-core") | .executable' | head -1)
    echo "Test binary: $TEST_BIN" | tee -a "$DEBUG_DIR/build_output.txt"

    if [ -n "$TEST_BIN" ] && [ -f "$TEST_BIN" ]; then
        # Disassemble functions containing "fastpfor" or "FastPFor"
        echo "=== Disassembly of FastPFor-related code ===" > "$DEBUG_DIR/disassembly.txt"
        objdump -d "$TEST_BIN" 2>/dev/null | grep -A 50 -i 'fastpfor\|FastPFor' | head -500 >> "$DEBUG_DIR/disassembly.txt" 2>&1 || echo "No fastpfor symbols found" >> "$DEBUG_DIR/disassembly.txt"

        # Check for AVX instructions in the binary
        echo "=== AVX instruction count in test binary ===" > "$DEBUG_DIR/avx_instructions.txt"
        objdump -d "$TEST_BIN" 2>/dev/null | grep -cE '\s(vmov|vadd|vperm|vxor|vpand|vpadd)' >> "$DEBUG_DIR/avx_instructions.txt" 2>&1 || echo "0" >> "$DEBUG_DIR/avx_instructions.txt"

        # List all ymm register usage (AVX 256-bit)
        echo "=== ymm register (AVX) usage count ===" >> "$DEBUG_DIR/avx_instructions.txt"
        objdump -d "$TEST_BIN" 2>/dev/null | grep -cE 'ymm[0-9]+' >> "$DEBUG_DIR/avx_instructions.txt" 2>&1 || echo "0" >> "$DEBUG_DIR/avx_instructions.txt"
    fi

    # Find and save CMake build logs from fastpfor
    echo "=== Looking for fastpfor CMake logs ===" | tee "$DEBUG_DIR/cmake_logs.txt"
    find target -name 'CMakeOutput.log' -o -name 'CMakeError.log' -o -name 'CMakeCache.txt' 2>/dev/null | head -10 | while read f; do
        echo "--- $f ---" >> "$DEBUG_DIR/cmake_logs.txt"
        cat "$f" >> "$DEBUG_DIR/cmake_logs.txt" 2>&1
        echo "" >> "$DEBUG_DIR/cmake_logs.txt"
    done

    # Extract key cmake settings (what compiler and flags cmake actually used)
    echo "=== CMake compiler/flags from cache ===" > "$DEBUG_DIR/cmake_compiler_flags.txt"
    CMAKE_CACHE_FILE=$(find target -name 'CMakeCache.txt' 2>/dev/null | head -1)
    if [ -n "$CMAKE_CACHE_FILE" ]; then
        grep -E '(CMAKE_CXX_COMPILER|CMAKE_C_COMPILER|CMAKE_CXX_FLAGS|MARCH|SUPPORT_AVX|SUPPORT_SSE)' "$CMAKE_CACHE_FILE" >> "$DEBUG_DIR/cmake_compiler_flags.txt" 2>&1 || true
    else
        echo "CMakeCache.txt not found - fastpfor may not be built yet" >> "$DEBUG_DIR/cmake_compiler_flags.txt"
    fi

    echo ""
    echo "=== Running the failing test ==="
    echo "Test: test_fastpfor_roundtrip::case_2_large"
    echo ""

    # Run with RUST_BACKTRACE=full
    export RUST_BACKTRACE=full

    # Try running with different verbosity levels and capture output
    echo "=== Test execution output ===" > "$DEBUG_DIR/test_output.txt"

    # First, try running normally
    set +e
    cargo test -p mlt-core --all-features \
        layer::v01::stream::tests::test_fastpfor_roundtrip::case_2_large \
        --include-ignored \
        -- --test-threads=1 --nocapture 2>&1 | tee -a "$DEBUG_DIR/test_output.txt"
    TEST_EXIT_CODE=$?
    set -e

    echo "" | tee -a "$DEBUG_DIR/test_output.txt"
    echo "Test exit code: $TEST_EXIT_CODE" | tee -a "$DEBUG_DIR/test_output.txt"

    # If test failed, try to get more info
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "" | tee -a "$DEBUG_DIR/test_output.txt"
        echo "=== Test FAILED - collecting additional debug info ===" | tee -a "$DEBUG_DIR/test_output.txt"

        # Try running with gdb if available
        if command -v gdb &> /dev/null && [ -n "$TEST_BIN" ]; then
            echo "=== Running with GDB to catch SIGILL ===" | tee "$DEBUG_DIR/gdb_output.txt"
            gdb -batch \
                -ex "set pagination off" \
                -ex "run layer::v01::stream::tests::test_fastpfor_roundtrip::case_2_large --include-ignored --test-threads=1" \
                -ex "bt full" \
                -ex "info registers" \
                -ex "x/10i \$pc-20" \
                -ex "x/10i \$pc" \
                --args "$TEST_BIN" 2>&1 | tee -a "$DEBUG_DIR/gdb_output.txt" || true
        else
            echo "GDB not available or test binary not found" | tee "$DEBUG_DIR/gdb_output.txt"
        fi

        # Try with lldb if available
        if command -v lldb &> /dev/null && [ -n "$TEST_BIN" ]; then
            echo "=== Running with LLDB ===" | tee "$DEBUG_DIR/lldb_output.txt"
            lldb -b \
                -o "run layer::v01::stream::tests::test_fastpfor_roundtrip::case_2_large --include-ignored --test-threads=1" \
                -o "bt all" \
                -o "register read" \
                -o "disassemble -p -c 20" \
                -- "$TEST_BIN" 2>&1 | tee -a "$DEBUG_DIR/lldb_output.txt" || true
        fi
    fi

    # Print summary
    echo ""
    echo "============================================================"
    echo "DEBUG ARTIFACTS SAVED TO: $DEBUG_DIR"
    echo "============================================================"
    ls -la "$DEBUG_DIR"
    echo ""

    # Print key info to stdout for easy viewing in CI logs
    echo "=== Key Environment Summary ==="
    cat "$ENV_FILE"
    echo ""
    echo "=== AVX Flags ==="
    cat "$DEBUG_DIR/avx_flags.txt"
    echo ""
    echo "=== AVX2 Basic Test Result ==="
    cat "$DEBUG_DIR/avx_test.txt"
    echo ""
    echo "=== VAES Test Result ==="
    cat "$DEBUG_DIR/vaes_test.txt"
    echo ""
    echo "=== VPCLMULQDQ Test Result ==="
    cat "$DEBUG_DIR/vpclmulqdq_test.txt"
    echo ""
    echo "=== CMake compiler/flags ==="
    cat "$DEBUG_DIR/cmake_compiler_flags.txt"
    echo ""

    # If the test failed, try rebuilding fastpfor with a safe march flag to check if that fixes it
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "=== Trying build with -march=x86-64-v3 wrapper (bypasses -march=native in fastpfor CMake) ==="
        echo "If this passes, -march=native is generating instructions not supported by the VM hypervisor"
        echo ""

        # Create a compiler wrapper that replaces -march=native with -march=x86-64-v3
        WRAPPER=/tmp/cxx-safe-march.sh
        cat > "$WRAPPER" << 'WRAPEOF'
    #!/bin/bash
    # Wrapper: replace -march=native with -march=x86-64-v3 to avoid SIGILL on AMD under Hyper-V
    # See: https://github.com/maplibre/maplibre-tile-spec/issues/1016
    args=()
    for arg in "$@"; do
        if [[ "$arg" == "-march=native" ]]; then
            args+=("-march=x86-64-v3")
        else
            args+=("$arg")
        fi
    done
    exec g++ "${args[@]}"
    WRAPEOF
        chmod +x "$WRAPPER"
        echo "Created compiler wrapper at $WRAPPER (replaces -march=native with -march=x86-64-v3)"

        echo "=== Safe-march test output ===" > "$DEBUG_DIR/safe_march_test.txt"
        set +e
        CXX="$WRAPPER" cargo test -p mlt-core \
            --target-dir "target/amd-avx-debug/safe-march-build" \
            --all-features \
            layer::v01::stream::tests::test_fastpfor_roundtrip::case_2_large \
            --include-ignored \
            -- --test-threads=1 --nocapture 2>&1 | tee -a "$DEBUG_DIR/safe_march_test.txt"
        SAFE_MARCH_EXIT=$?
        set -e

        echo "Safe-march test exit code: $SAFE_MARCH_EXIT" | tee -a "$DEBUG_DIR/safe_march_test.txt"

        if [ $SAFE_MARCH_EXIT -eq 0 ]; then
            echo ""
            echo "*** SAFE-MARCH TEST PASSED! ***"
            echo "*** Root cause confirmed: -march=native generates instructions unsupported by VM ***"
            echo "*** Fix: update fastpfor CMakeLists.txt to allow disabling -march=native ***"
        else
            echo ""
            echo "Safe-march test also failed - the issue may not be -march=native related"
        fi
    fi

    # Exit with the test result
    exit $TEST_EXIT_CODE

# Run linting as expected by CI
ci-lint: ci-env-info test-fmt clippy
    {{just}} assert-git-is-clean  # call it explicitly to avoid just optimizing it to just one call

# Run all tests as expected by CI
ci-test: ci-env-info (just 'java::gen-snippets') test
    {{just}} assert-git-is-clean  # call it explicitly to avoid just optimizing it to just one call

# Run minimal subset of tests to ensure compatibility with MSRV
ci-test-msrv: ci-env-info
    #!/usr/bin/env bash
    set -euo pipefail
    # For MSRV, relax all warnings to avoid different warnings per rust version
    unset CI RUSTFLAGS RUSTDOCFLAGS RUST_BACKTRACE
    # Pre-install cargo-hack to avoid CI binstall issues
    {{just}} cargo-install cargo-hack
    {{just}} rust::ci-check
    {{just}} java::gen-snippets
    {{just}} rust::test
    {{just}} assert-git-is-clean

# Build the layer target (installs `cargo fuzz`)
ci-fuzz-build target: (cargo-fuzz 'build' target '--target' 'x86_64-unknown-linux-gnu')

# Fuzz the target for 90 seconds (installs `cargo fuzz`)
ci-fuzz-run target: (cargo-fuzz 'run' target '--target' 'x86_64-unknown-linux-gnu' '--' '-max_total_time=90' '--max_len=300')

# Build and package release binary for CI
ci-build-release target artifact_name:
    #!/usr/bin/env bash
    set -euo pipefail
    rustup target add {{target}}
    cargo build --release --target {{target}} --package mlt --bin mlt
    cd target/{{target}}/release
    if [ -f mlt.exe ]; then
        7z a ../../../../{{artifact_name}} mlt.exe
    else
        tar -czf ../../../../{{artifact_name}} mlt
    fi

# Output version and tag for GITHUB_OUTPUT (append with >> $GITHUB_OUTPUT)
ci-get-release-info package='mlt':
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$({{just}} rust::get-crate-field version {{package}})
    echo "version=$VERSION"
    echo "tag=rust-{{package}}-v$VERSION"

# Clean all build artifacts
clean:
    cargo clean
    rm -f Cargo.lock

# Run cargo clippy to lint the code
clippy *args:
    cargo clippy --workspace --all-targets --all-features {{args}}
    cd mlt-core/fuzz && cargo clippy --all-features {{args}}

# Generate code coverage report. Installs `cargo llvm-cov`
coverage *args='--no-clean --open': (just 'cargo-install' 'cargo-llvm-cov')
    cargo llvm-cov --all-targets --all-features --include-build-script {{args}}

# Build and open code documentation
docs *args='--open':
    DOCS_RS=1 cargo doc --no-deps {{args}} --all-features --package mlt-core

alias f := fmt
# Reformat code. If nightly is available, use it for better results.
fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    if (rustup toolchain list | grep nightly && rustup component list --toolchain nightly | grep rustfmt) &> /dev/null; then
        echo 'Reformatting Rust code using nightly Rust fmt to sort imports'
        cargo +nightly fmt --all -- --config imports_granularity=Module,group_imports=StdExternalCrate
    else
        echo 'Reformatting Rust with the stable cargo fmt.  Install nightly with `rustup install nightly` for better results'
        cargo fmt --all
    fi

# Reformat all Cargo.toml files (installs `cargo sort`)
fmt-toml *args: (just 'cargo-install' 'cargo-sort')
    cargo sort --workspace --grouped {{args}}

# Fuzz the layer target (installs `cargo fuzz`)
fuzz *args: (cargo-fuzz 'run' 'layer' '--' args)

# Minimize the size of the fuzzing corpus (installs `cargo fuzz`)
fuzz-cmin file: (cargo-fuzz 'tmin' 'layer' file)

# Minimize the crash in file (installs `cargo fuzz`)
fuzz-tmin file: (cargo-fuzz 'tmin' 'layer' file)

# Get any package's field from the metadata
get-crate-field field package=main_crate: (just 'assert-cmd' 'jq')
    cargo metadata --format-version 1 | jq -e -r '.packages | map(select(.name == "{{package}}")) | first | .{{field}} | select(. != null)'

# Get the minimum supported Rust version (MSRV) for the crate
get-msrv package=main_crate: (get-crate-field 'rust_version' package)

# Run linting (clippy + format check)
lint: clippy test-fmt

# Find the minimum supported Rust version and update Cargo.toml
msrv: (just 'cargo-install' 'cargo-msrv')
    cargo msrv find --write-msrv --ignore-lockfile --component rustfmt -- {{just}} rust::ci-test-msrv

# Build mlt-py and launch a Python interpreter with mlt pre-imported
[working-directory: 'mlt-py']
py: (just 'cargo-install' 'maturin')
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -d ".venv" ]]; then
        echo "Please make sure to create your virtual environment by running 'python -m venv .venv'"
        exit 1
    fi
    source .venv/bin/activate
    maturin develop
    python3 -i -c "import maplibre_tiles; print('maplibre_tiles ready — try: maplibre_tiles.decode_mlt(data)')"

py-publish:
    docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

# Run cargo-release
release *args='': (just 'cargo-install' 'release-plz')
    release-plz {{args}}

# Check semver compatibility with prior published version
semver *args: (just 'cargo-install' 'cargo-semver-checks')
    cargo semver-checks --all-features {{args}}

# Regenerate the Python type stub for mlt-py
sync-pyo3-stubs:
    cargo run -p mlt-py --bin stub_gen --features abi3

# Sync the version field in mlt-py/pyproject.toml to match mlt-py/Cargo.toml
sync-pyo3-version:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$({{just}} rust::get-crate-field version mlt-py)
    PYPROJECT="mlt-py/pyproject.toml"
    sed -i "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT"
    echo "Synced $PYPROJECT to version $VERSION"

alias t := test
# Run all tests
test:
    cargo test --all-features --all-targets
    {{just}} rust::generate-synthetic-mlts
    {{just}} rust::test-doc

# Test documentation generation
test-doc: (docs '--document-private-items')
    cargo test --doc --all-features

# Test code formatting
test-fmt: (fmt-toml '--check' '--check-format')
    cargo fmt --all -- --check

# Test workspace publishing with --dry-run
test-publish:
    cargo publish --workspace --dry-run

# Find unused dependencies
udeps: (just 'cargo-install' 'cargo-udeps')
    cargo +nightly udeps --all-targets --all-features

# Update all dependencies, including breaking changes
update:
    cargo +nightly -Z unstable-options update --breaking
    cargo update

# Generate synthetic .mlt files and ensure there are no duplicates
generate-synthetic-mlts:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf ../test/synthetic/0x01-rust
    cargo run -p mlt-synthetics
    {{just}} rust::validate-synthetic
    {{just}} _assert-all-mlt-files-different test/synthetic/0x01-rust
    {{just}} rust::_verify-synthetic-equality
    {{just}} rust::_assert-synthetic-mlt-identity

# Create .dump files for all .mlt files in test/synthetic for all languages
generate-synthetic-dumps: (just 'cargo-install' 'fd' 'fd-find') build
    fd . ../test/synthetic -e mlt -x sh -c 'target/debug/mlt dump {} > {.}.dump'

# Validate every .mlt in test/synthetic against corresponding .json (GeoJSON)
validate-synthetic:
    {{just}} mlt ls --validate-to-json --details basic ../test/synthetic/

# Assert each .mlt in compare_dir exists in ref_dir and is binary-identical. Skip '_rust' files.
_assert-synthetic-mlt-identity ref_dir='../test/synthetic/0x01' compare_dir='../test/synthetic/0x01-rust':
    #!/usr/bin/env bash
    set -euo pipefail
    # Ensure compare_dir exists.
    if [[ ! -d "{{compare_dir}}" ]]; then
        echo "::error::Compare directory {{compare_dir}} does not exist"
        exit 1
    fi

    # Expand .mlt files; fail if none are found to avoid masking missing/failed generation.
    shopt -s nullglob
    mlt_files=( "{{compare_dir}}"/*.mlt )
    shopt -u nullglob
    if (( ${#mlt_files[@]} == 0 )); then
        echo "::error::No .mlt files found in {{compare_dir}}"
        exit 1
    fi

    for mlt in "${mlt_files[@]}"; do
        base=$(basename "$mlt")
        if [[ "$base" == *"-rust.mlt" ]]; then
            continue
        fi
        ref="{{ref_dir}}/$base"
        if [[ ! -f "$ref" ]]; then
            echo "::error::File $mlt has no counterpart in {{ref_dir}}"
            exit 1
        elif ! cmp -s "$ref" "$mlt"; then
            echo "::error::Binary difference: $base"
            exit 1
        fi
    done
    echo "All {{compare_dir}}/*.mlt files have identical counterparts in {{ref_dir}} (except *-rust.mlt)."

# checks that the contents of two synthetic directories are equal
_verify-synthetic-equality: (just 'cargo-install' 'delta' 'git-delta')
    #!/usr/bin/env bash
    set -euo pipefail

    # `realpath --relative-to-...` doesn't work on MacOS
    # https://unix.stackexchange.com/a/655825/312244
    posix_relpath() {
        set -- "${1%/}/" "${2%/}/" ''
        while [ "$1" ] && [ "$2" = "${2#"$1"}" ]
        do
            set -- "${1%/?*/}/" "$2" "../$3"
        done
        rel="${3}${2#"$1"}"
        [ "${rel#/}" ] && rel="${rel%/}" || rel="${rel:-.}"
        printf '%s\n' "$rel"
    }

    DIR_A="../test/synthetic/0x01"
    DIR_B="../test/synthetic/0x01-rust"

    BIN="cargo run --bin mlt --quiet --"
    find "$DIR_A" -type f -iname "*.mlt" | while read -r file_a; do
      rel_path="$(posix_relpath "$DIR_A" "$file_a")"
      file_b="$DIR_B/$rel_path"
      file_b_rust="${file_b%.mlt}-rust.mlt"

      if [[ -f "$file_b" ]]; then
        if ! cmp -s "$file_a" "$file_b"; then
          echo "Differences: $rel_path"
          echo ""
          diff -U20 \
            --label "$file_a" \
            <($BIN dump "$file_a") \
            --label "$file_b" \
            <($BIN dump "$file_b") \
            | delta --side-by-side --file-modified-label "mlt dump difference"
        fi
      elif [[ -f "$file_b_rust" ]]; then
        diff -U20 \
          --label "$file_a" \
          <($BIN decode "$file_a") \
          --label "$file_b_rust" \
          <($BIN decode "$file_b_rust") \
          | delta --side-by-side --file-modified-label "mlt decoded difference"
      else
        echo "Missing in $DIR_B but present in $DIR_A: $rel_path"
      fi
    done


[private]
[working-directory: 'mlt-core']
cargo-fuzz *args:  (just 'cargo-install' 'cargo-fuzz')
    cargo +nightly fuzz {{args}}
