

[external]
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_local_idj(i32 0) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #3
LcallBD
B
	full_text5
3
1%8 = tail call i64 @_Z14get_local_sizej(i32 0) #3
,shlB%
#
	full_text

%9 = shl i64 %8, 1
"i64B

	full_text


i64 %8
.mulB'
%
	full_text

%10 = mul i64 %9, %7
"i64B

	full_text


i64 %9
"i64B

	full_text


i64 %7
6andB/
-
	full_text 

%11 = and i64 %5, 4294967295
"i64B

	full_text


i64 %5
/addB(
&
	full_text

%12 = add i64 %10, %5
#i64B

	full_text
	
i64 %10
"i64B

	full_text


i64 %5
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
McallBE
C
	full_text6
4
2%14 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
/mulB(
&
	full_text

%15 = mul i64 %9, %14
"i64B

	full_text


i64 %9
#i64B

	full_text
	
i64 %14
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
5truncB,
*
	full_text

%17 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
ZgetelementptrBI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %2, i64 %11
#i64B

	full_text
	
i64 %11
SstoreBJ
H
	full_text;
9
7store float 0.000000e+00, float* %18, align 4, !tbaa !8
)float*B

	full_text


float* %18
5icmpB-
+
	full_text

%19 = icmp ult i32 %13, %3
#i32B

	full_text
	
i32 %13
8brB2
0
	full_text#
!
br i1 %19, label %20, label %35
!i1B

	full_text


i1 %19
'br8B

	full_text

br label %21
Ophi8BF
D
	full_text7
5
3%22 = phi float [ %32, %21 ], [ 0.000000e+00, %20 ]
)float8B

	full_text

	float %32
Dphi8B;
9
	full_text,
*
(%23 = phi i32 [ %33, %21 ], [ %13, %20 ]
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %13
6zext8B,
*
	full_text

%24 = zext i32 %23 to i64
%i328B

	full_text
	
i32 %23
\getelementptr8BI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %0, i64 %24
%i648B

	full_text
	
i64 %24
Lload8BB
@
	full_text3
1
/%26 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
2add8B)
'
	full_text

%27 = add i32 %23, %17
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %17
6zext8B,
*
	full_text

%28 = zext i32 %27 to i64
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
6fadd8B,
*
	full_text

%32 = fadd float %22, %31
)float8B

	full_text

	float %22
)float8B

	full_text

	float %31
Lstore8BA
?
	full_text2
0
.store float %32, float* %18, align 4, !tbaa !8
)float8B

	full_text

	float %32
+float*8B

	full_text


float* %18
2add8B)
'
	full_text

%33 = add i32 %23, %16
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %16
7icmp8B-
+
	full_text

%34 = icmp ult i32 %33, %3
%i328B

	full_text
	
i32 %33
:br8B2
0
	full_text#
!
br i1 %34, label %21, label %35
#i18B

	full_text


i1 %34
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2lshr8B(
&
	full_text

%36 = lshr i32 %17, 1
%i328B

	full_text
	
i32 %17
5icmp8B+
)
	full_text

%37 = icmp eq i32 %36, 0
%i328B

	full_text
	
i32 %36
:br8B2
0
	full_text#
!
br i1 %37, label %39, label %38
#i18B

	full_text


i1 %37
'br8B

	full_text

br label %41
4icmp8B*
(
	full_text

%40 = icmp eq i32 %6, 0
$i328B

	full_text


i32 %6
:br8B2
0
	full_text#
!
br i1 %40, label %54, label %59
#i18B

	full_text


i1 %40
Dphi8B;
9
	full_text,
*
(%42 = phi i32 [ %52, %51 ], [ %36, %38 ]
%i328B

	full_text
	
i32 %52
%i328B

	full_text
	
i32 %36
7icmp8B-
+
	full_text

%43 = icmp ugt i32 %42, %6
%i328B

	full_text
	
i32 %42
$i328B

	full_text


i32 %6
:br8B2
0
	full_text#
!
br i1 %43, label %44, label %51
#i18B

	full_text


i1 %43
1add8B(
&
	full_text

%45 = add i32 %42, %6
%i328B

	full_text
	
i32 %42
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%46 = zext i32 %45 to i64
%i328B

	full_text
	
i32 %45
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %2, i64 %46
%i648B

	full_text
	
i64 %46
Lload8BB
@
	full_text3
1
/%48 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
Lload8BB
@
	full_text3
1
/%49 = load float, float* %18, align 4, !tbaa !8
+float*8B

	full_text


float* %18
6fadd8B,
*
	full_text

%50 = fadd float %48, %49
)float8B

	full_text

	float %48
)float8B

	full_text

	float %49
Lstore8BA
?
	full_text2
0
.store float %50, float* %18, align 4, !tbaa !8
)float8B

	full_text

	float %50
+float*8B

	full_text


float* %18
'br8B

	full_text

br label %51
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2lshr8B(
&
	full_text

%52 = lshr i32 %42, 1
%i328B

	full_text
	
i32 %42
5icmp8B+
)
	full_text

%53 = icmp eq i32 %52, 0
%i328B

	full_text
	
i32 %52
:br8B2
0
	full_text#
!
br i1 %53, label %39, label %41
#i18B

	full_text


i1 %53
?bitcast8	B2
0
	full_text#
!
%55 = bitcast float* %2 to i32*
Hload8	B>
<
	full_text/
-
+%56 = load i32, i32* %55, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %55
[getelementptr8	BH
F
	full_text9
7
5%57 = getelementptr inbounds float, float* %1, i64 %7
$i648	B

	full_text


i64 %7
@bitcast8	B3
1
	full_text$
"
 %58 = bitcast float* %57 to i32*
+float*8	B

	full_text


float* %57
Hstore8	B=
;
	full_text.
,
*store i32 %56, i32* %58, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %56
'i32*8	B

	full_text


i32* %58
'br8	B

	full_text

br label %59
$ret8
B

	full_text


ret void
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
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
i32 0
#i648B

	full_text	

i64 1
,i648B!

	full_text

i64 4294967295
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 1       	 
                        !  $ ## %& %' %% () (( *+ ** ,- ,, ./ .0 .. 12 11 34 33 56 55 78 79 77 :; :< :: => =? == @A @B @@ CD CC EF EG HI HH JK JJ LM LP OO QR QT SU SS VW VX VV YZ Y\ [] [[ ^_ ^^ `a `` bc bb de dd fg fh ff ij ik ii lm no nn pq pp rs rt uv uu wx ww yz yy {| {} {{ ~? ? `? t	? 	? C? *? 3? w   	 
            !: $@ & '% )( +* -% / 0. 21 43 6, 85 9# ;7 <: > ?% A B@ DC F IH KJ M PO Rn TH US W XV ZS \ ][ _^ a` c eb gd hf j kS on qp st v xw zu |y }  "  G" #L OL NE #E GQ tQ N S~ Y [Y ml mr Or S  ?? ?? ?? ?? ?? ??  ?? G ?? Gm ?? m ??  ?? ? ? ? ? 	? J	? O	? p	? 	? ? 	? #? G	? H? m	? n"
reduce"
_Z12get_local_idj"
_Z12get_group_idj"
_Z14get_local_sizej"
_Z14get_num_groupsj"
_Z7barrierj*?
shoc-1.1.5-Reduction-reduce.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???

wgsize_log1p
a?sA

devmap_label


wgsize
?
 
transfer_bytes_log1p
a?sA