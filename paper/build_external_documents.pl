#!/usr/bin/env perl
# Bootstrap reciprocal xr references without recursive latexmkrc execution.
# Each child latexmk reads its ordinary .fls/.fdb_latexmk dependencies, so
# changes in an included .tex, .bib, .sty, or figure are handled automatically.
# The fixed-point loop also refreshes imported labels after page/number changes.
# References: https://www.overleaf.com/learn/how-to/Cross_referencing_with_the_xr_package_in_Overleaf

use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use File::Copy qw(copy);
use File::Path qw(make_path);

my $build = 'nmkc-build';
my @documents = ('main', 'supplement');
make_path($build);
make_path("$build/$_") for @documents;

sub contents {
    my ($path) = @_;
    return '' unless -f $path;
    open my $fh, '<:raw', $path or die "Cannot read $path: $!\n";
    local $/;
    return <$fh>;
}

sub label_state {
    return sha256_hex(join("\0", map { contents("$_.aux") } @documents));
}

sub compile_document {
    my ($document) = @_;
    # Distinct child job names also prevent an old root main.bbl/main.aux
    # from being mistaken for this build's bibliography or auxiliary file.
    my $job = "nmkc-$document";
    my $log = "$build/$document.build.log";
    my $pid = fork();
    die "Cannot start a build process: $!\n" unless defined $pid;
    if ($pid == 0) {
        open STDOUT, '>', $log or die "Cannot write $log: $!\n";
        open STDERR, '>&', \*STDOUT or die "Cannot redirect build errors: $!\n";
        # -norc prevents this child from loading the outer latexmkrc again.
        exec('latexmk', '-norc', '-pdf', '-interaction=nonstopmode',
             '-halt-on-error', '-file-line-error',
             "-jobname=$job", "-outdir=$build/$document", "$document.tex");
        die "Cannot execute latexmk: $!\n";
    }
    waitpid($pid, 0);
    my $status = $?;
    if ($status != 0) {
        my @lines = split /\n/, contents($log);
        my $start = @lines > 35 ? @lines - 35 : 0;
        print STDERR join("\n", @lines[$start .. $#lines]), "\n";
        die "Cannot compile $document.tex; full details: $log\n";
    }
    my $aux = "$build/$document/$job.aux";
    die "The $document build did not produce $aux\n" unless -s $aux;
    # xr searches beside the .tex sources, independently of the editor job name.
    # Each document owns its bibliography.  Older xr versions also import
    # \bibcite, causing duplicate citation labels.  Export only \newlabel
    # records; all sections in these two roots use \input, so their labels
    # are written directly in the root aux file, without nested aux files.
    my @labels = contents($aux) =~ /^(\\newlabel\{[^\n]*)/mg;
    # hyperref stores the external PDF in its fifth label field.  Naming it
    # here keeps reciprocal links external even with older versions of xr;
    # otherwise identical anchor names can incorrectly link inside this PDF.
    s/\{\}\}\s*$/{$document.pdf}}/ for @labels;
    my $export = "% Generated cross-reference labels; rebuild instead of editing.\n"
               . join("\n", @labels) . "\n";
    if (contents("$document.aux") ne $export) {
        open my $out, '>', "$document.aux" or die "Cannot export $document labels: $!\n";
        print {$out} $export;
        close $out;
    }
}

my $settled = 0;
for my $round (1 .. 6) {
    my $before = label_state();
    compile_document($_) for @documents;
    if (label_state() eq $before) {
        $settled = 1;
        last;
    }
}
die "The two documents' labels did not settle after six rounds.\n" unless $settled;

for my $document (@documents) {
    my $job = "nmkc-$document";
    my $log = contents("$build/$document/$job.log");
    if ($log =~ /(?:There were undefined references|There were undefined citations)/) {
        die "Unresolved references remain in $document.tex; inspect $build/$document/$job.log.\n";
    }
    copy("$build/$document/$job.pdf", "$document.pdf")
        or die "Cannot export $document.pdf: $!\n";
}
print "NMKC: main.pdf and supplement.pdf built; reciprocal references settled.\n";
