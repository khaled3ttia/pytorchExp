

[external]
.sdivB&
$
	full_text

%5 = sdiv i32 %2, 4
2sextB*
(
	full_text

%6 = sext i32 %5 to i64
"i32B

	full_text


i32 %5
LcallBD
B
	full_text5
3
1%7 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
/udivB'
%
	full_text

%8 = udiv i64 %6, %7
"i64B

	full_text


i64 %6
"i64B

	full_text


i64 %7
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
-shlB&
$
	full_text

%10 = shl i32 %9, 2
"i32B

	full_text


i32 %9
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_group_idj(i32 0) #3
4sextB,
*
	full_text

%12 = sext i32 %10 to i64
#i32B

	full_text
	
i32 %10
0mulB)
'
	full_text

%13 = mul i64 %11, %12
#i64B

	full_text
	
i64 %11
#i64B

	full_text
	
i64 %12
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
.addB'
%
	full_text

%15 = add i64 %7, -1
"i64B

	full_text


i64 %7
5icmpB-
+
	full_text

%16 = icmp eq i64 %11, %15
#i64B

	full_text
	
i64 %11
#i64B

	full_text
	
i64 %15
4addB-
+
	full_text

%17 = add nsw i32 %10, %14
#i32B

	full_text
	
i32 %10
#i32B

	full_text
	
i32 %14
AselectB7
5
	full_text(
&
$%18 = select i1 %16, i32 %2, i32 %17
!i1B

	full_text


i1 %16
#i32B

	full_text
	
i32 %17
KcallBC
A
	full_text4
2
0%19 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%20 = trunc i64 %19 to i32
#i64B

	full_text
	
i64 %19
4addB-
+
	full_text

%21 = add nsw i32 %14, %20
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %20
6icmpB.
,
	full_text

%22 = icmp slt i32 %21, %18
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %18
McallBE
C
	full_text6
4
2%23 = tail call i64 @_Z14get_local_sizej(i32 0) #3
8brB2
0
	full_text#
!
br i1 %22, label %24, label %35
!i1B

	full_text


i1 %22
'br8B

	full_text

br label %25
Ophi8BF
D
	full_text7
5
3%26 = phi float [ %31, %25 ], [ 0.000000e+00, %24 ]
)float8B

	full_text

	float %31
Dphi8B;
9
	full_text,
*
(%27 = phi i32 [ %33, %25 ], [ %21, %24 ]
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %21
6sext8B,
*
	full_text

%28 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %0, i64 %28
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !8
+float*8B

	full_text


float* %29
6fadd8B,
*
	full_text

%31 = fadd float %26, %30
)float8B

	full_text

	float %26
)float8B

	full_text

	float %30
2add8B)
'
	full_text

%32 = add i64 %23, %28
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %28
8trunc8B-
+
	full_text

%33 = trunc i64 %32 to i32
%i648B

	full_text
	
i64 %32
8icmp8B.
,
	full_text

%34 = icmp sgt i32 %18, %33
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %33
:br8B2
0
	full_text#
!
br i1 %34, label %25, label %35
#i18B

	full_text


i1 %34
Nphi8BE
C
	full_text6
4
2%36 = phi float [ 0.000000e+00, %4 ], [ %31, %25 ]
)float8B

	full_text

	float %31
1shl8B(
&
	full_text

%37 = shl i64 %19, 32
%i648B

	full_text
	
i64 %19
9ashr8B/
-
	full_text 

%38 = ashr exact i64 %37, 32
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %3, i64 %38
%i648B

	full_text
	
i64 %38
Lstore8BA
?
	full_text2
0
.store float %36, float* %39, align 4, !tbaa !8
)float8B

	full_text

	float %36
+float*8B

	full_text


float* %39
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2lshr8B(
&
	full_text

%40 = lshr i64 %23, 1
%i648B

	full_text
	
i64 %23
8trunc8B-
+
	full_text

%41 = trunc i64 %40 to i32
%i648B

	full_text
	
i64 %40
5icmp8B+
)
	full_text

%42 = icmp eq i32 %41, 0
%i328B

	full_text
	
i32 %41
:br8B2
0
	full_text#
!
br i1 %42, label %44, label %43
#i18B

	full_text


i1 %42
'br8B

	full_text

br label %46
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
5icmp8B+
)
	full_text

%45 = icmp eq i32 %20, 0
%i328B

	full_text
	
i32 %20
:br8B2
0
	full_text#
!
br i1 %45, label %59, label %64
#i18B

	full_text


i1 %45
Dphi8B;
9
	full_text,
*
(%47 = phi i32 [ %57, %56 ], [ %41, %43 ]
%i328B

	full_text
	
i32 %57
%i328B

	full_text
	
i32 %41
8icmp8B.
,
	full_text

%48 = icmp ugt i32 %47, %20
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %20
:br8B2
0
	full_text#
!
br i1 %48, label %49, label %56
#i18B

	full_text


i1 %48
2add8B)
'
	full_text

%50 = add i32 %47, %20
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %20
6zext8B,
*
	full_text

%51 = zext i32 %50 to i64
%i328B

	full_text
	
i32 %50
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %3, i64 %51
%i648B

	full_text
	
i64 %51
Lload8BB
@
	full_text3
1
/%53 = load float, float* %52, align 4, !tbaa !8
+float*8B

	full_text


float* %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %39, align 4, !tbaa !8
+float*8B

	full_text


float* %39
6fadd8B,
*
	full_text

%55 = fadd float %53, %54
)float8B

	full_text

	float %53
)float8B

	full_text

	float %54
Lstore8BA
?
	full_text2
0
.store float %55, float* %39, align 4, !tbaa !8
)float8B

	full_text

	float %55
+float*8B

	full_text


float* %39
'br8B

	full_text

br label %56
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2lshr8B(
&
	full_text

%57 = lshr i32 %47, 1
%i328B

	full_text
	
i32 %47
5icmp8B+
)
	full_text

%58 = icmp eq i32 %57, 0
%i328B

	full_text
	
i32 %57
:br8B2
0
	full_text#
!
br i1 %58, label %44, label %46
#i18B

	full_text


i1 %58
?bitcast8	B2
0
	full_text#
!
%60 = bitcast float* %3 to i32*
Hload8	B>
<
	full_text/
-
+%61 = load i32, i32* %60, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %60
\getelementptr8	BI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %1, i64 %11
%i648	B

	full_text
	
i64 %11
@bitcast8	B3
1
	full_text$
"
 %63 = bitcast float* %62 to i32*
+float*8	B

	full_text


float* %62
Hstore8	B=
;
	full_text.
,
*store i32 %61, i32* %63, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %61
'i32*8	B

	full_text


i32* %63
'br8	B

	full_text

br label %64
$ret8
B

	full_text


ret void
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %1
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 4
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -1
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32       	  
 

                      !    "# "$ "" %& %' %% (( )* )- ,, ./ .0 .. 12 11 34 33 56 55 78 79 77 :; :< :: => == ?@ ?A ?? BC BE DD FG FF HI HH JK JJ LM LN LL OO PQ PP RS RR TU TT VW VY Z[ ZZ \] \_ ^` ^^ ab ac aa de dg fh ff ij ii kl kk mn mm op oo qr qs qq tu tv tt wx yz yy {| {{ }~ } ?? ?? ?
? ?? ?? ?? ?? ?
? ?? ?? J? k? ? 3? 	? ? ?    	 
       
     ! #  $" & '% *7 -= /" 0. 21 43 6, 85 9( ;1 <: > @= A? C7 E GF IH KD MJ N( QP SR UT W  [Z ]y _R `^ b  ca e^ g  hf ji lk nJ pm ro sq uJ v^ zy |{ ~ ? ?? ?? ?? ?) +) D+ ,V YV XB ,B D\ \ ?X ^? ?d fd xw x} Y} ^ ?? ? ?? ?? ?? ?? ?? Y ?? YO ?? O ?? ( ?? (x ?? x ?? 	? 
	? ? O? Y? x	? y	? P	? 	? ,? D? ? ? ? (	? T	? Z	? {	? F	? H"
reduce"
_Z14get_num_groupsj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z7barrierj*?
shoc-1.1.5-Scan-reduce.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
cA

wgsize_log1p
cA

transfer_bytes
???

devmap_label


wgsize
?