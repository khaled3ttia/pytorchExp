

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #2
,addB%
#
	full_text

%7 = add i64 %6, 1
"i64B

	full_text


i64 %6
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #2
-addB&
$
	full_text

%10 = add i64 %9, 1
"i64B

	full_text


i64 %9
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #2
.addB'
%
	full_text

%12 = add i64 %11, 1
#i64B

	full_text
	
i64 %11
2addB+
)
	full_text

%13 = add nsw i32 %4, -2
5icmpB-
+
	full_text

%14 = icmp slt i32 %13, %8
#i32B

	full_text
	
i32 %13
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %14, label %57, label %15
!i1B

	full_text


i1 %14
8trunc8B-
+
	full_text

%16 = trunc i64 %12 to i32
%i648B

	full_text
	
i64 %12
8trunc8B-
+
	full_text

%17 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
4add8B+
)
	full_text

%18 = add nsw i32 %3, -2
8icmp8B.
,
	full_text

%19 = icmp slt i32 %18, %17
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %17
4add8B+
)
	full_text

%20 = add nsw i32 %2, -2
8icmp8B.
,
	full_text

%21 = icmp slt i32 %20, %16
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %16
/or8B'
%
	full_text

%22 = or i1 %19, %21
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %21
:br8B2
0
	full_text#
!
br i1 %22, label %57, label %23
#i18B

	full_text


i1 %22
Wbitcast8BJ
H
	full_text;
9
7%24 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%25 = bitcast double* %1 to [65 x [65 x [5 x double]]]*
0shl8B'
%
	full_text

%26 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%27 = ashr exact i64 %26, 32
%i648B

	full_text
	
i64 %26
1shl8B(
&
	full_text

%28 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
1shl8B(
&
	full_text

%30 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%31 = ashr exact i64 %30, 32
%i648B

	full_text
	
i64 %30
¢getelementptr8Bé
ã
	full_text~
|
z%32 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %24, i64 %27, i64 %29, i64 %31, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%33 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
¢getelementptr8Bé
ã
	full_text~
|
z%34 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %25, i64 %27, i64 %29, i64 %31, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
7fadd8B-
+
	full_text

%36 = fadd double %33, %35
+double8B

	full_text


double %33
+double8B

	full_text


double %35
Nstore8BC
A
	full_text4
2
0store double %36, double* %32, align 8, !tbaa !8
+double8B

	full_text


double %36
-double*8B

	full_text

double* %32
¢getelementptr8Bé
ã
	full_text~
|
z%37 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %24, i64 %27, i64 %29, i64 %31, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%38 = load double, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
¢getelementptr8Bé
ã
	full_text~
|
z%39 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %25, i64 %27, i64 %29, i64 %31, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
7fadd8B-
+
	full_text

%41 = fadd double %38, %40
+double8B

	full_text


double %38
+double8B

	full_text


double %40
Nstore8BC
A
	full_text4
2
0store double %41, double* %37, align 8, !tbaa !8
+double8B

	full_text


double %41
-double*8B

	full_text

double* %37
¢getelementptr8Bé
ã
	full_text~
|
z%42 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %24, i64 %27, i64 %29, i64 %31, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
¢getelementptr8Bé
ã
	full_text~
|
z%44 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %25, i64 %27, i64 %29, i64 %31, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
7fadd8B-
+
	full_text

%46 = fadd double %43, %45
+double8B

	full_text


double %43
+double8B

	full_text


double %45
Nstore8BC
A
	full_text4
2
0store double %46, double* %42, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %42
¢getelementptr8Bé
ã
	full_text~
|
z%47 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %24, i64 %27, i64 %29, i64 %31, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
¢getelementptr8Bé
ã
	full_text~
|
z%49 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %25, i64 %27, i64 %29, i64 %31, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%50 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
7fadd8B-
+
	full_text

%51 = fadd double %48, %50
+double8B

	full_text


double %48
+double8B

	full_text


double %50
Nstore8BC
A
	full_text4
2
0store double %51, double* %47, align 8, !tbaa !8
+double8B

	full_text


double %51
-double*8B

	full_text

double* %47
¢getelementptr8Bé
ã
	full_text~
|
z%52 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %24, i64 %27, i64 %29, i64 %31, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
¢getelementptr8Bé
ã
	full_text~
|
z%54 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %25, i64 %27, i64 %29, i64 %31, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %31
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
7fadd8B-
+
	full_text

%56 = fadd double %53, %55
+double8B

	full_text


double %53
+double8B

	full_text


double %55
Nstore8BC
A
	full_text4
2
0store double %56, double* %52, align 8, !tbaa !8
+double8B

	full_text


double %56
-double*8B

	full_text

double* %52
'br8B

	full_text

br label %57
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %4
-; undefined function B

	full_text

 
$i328B

	full_text


i32 -2
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 2        		 
 

                      !" !# $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 13 14 15 11 67 66 89 8: 8; 8< 88 => == ?@ ?A ?? BC BD BB EF EG EH EI EE JK JJ LM LN LO LP LL QR QQ ST SU SS VW VX VV YZ Y[ Y\ Y] YY ^_ ^^ `a `b `c `d `` ef ee gh gi gg jk jl jj mn mo mp mq mm rs rr tu tv tw tx tt yz yy {| {} {{ ~ ~	Ä ~~ ÅÇ Å
É Å
Ñ Å
Ö ÅÅ Üá ÜÜ àâ à
ä à
ã à
å àà çé çç èê è
ë èè íì í
î íí ïó ò #ô ö $õ    	    
          " &% ( *) ,
 .- 0# 2' 3+ 4/ 51 7$ 9' :+ ;/ <8 >6 @= A? C1 D# F' G+ H/ IE K$ M' N+ O/ PL RJ TQ US WE X# Z' [+ \/ ]Y _$ a' b+ c/ d` f^ he ig kY l# n' o+ p/ qm s$ u' v+ w/ xt zr |y }{ m Ä# Ç' É+ Ñ/ ÖÅ á$ â' ä+ ã/ åà éÜ êç ëè ìÅ î ñ ! ñ! #ï ñ ñ úú úú 	 úú 	 úú 	ù 	ù 	ù û 	ü %	ü '	ü )	ü +	ü -	ü /	† m	† t
° Å
° à	¢ Y	¢ `£ 		§ 1	§ 8	• 	• 	• 
	• E	• L¶ "
add"
_Z13get_global_idj*á
npb-T-add_A.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
ùÆúA

wgsize
>
 
transfer_bytes_log1p
ùÆúA

transfer_bytes	
ÿîÁò

devmap_label
