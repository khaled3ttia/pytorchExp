

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
3icmpB+
)
	full_text

%9 = icmp ult i32 %8, %3
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %21
 i1B

	full_text	

i1 %9
Mcall8BC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #4
8trunc8B-
+
	full_text

%12 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
7icmp8B-
+
	full_text

%13 = icmp ult i32 %12, %2
%i328B

	full_text
	
i32 %12
\getelementptr8BI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %5, i64 %11
%i648B

	full_text
	
i64 %11
5icmp8B+
)
	full_text

%15 = icmp eq i64 %11, 0
%i648B

	full_text
	
i64 %11
?bitcast8B2
0
	full_text#
!
%16 = bitcast float* %5 to i32*
Ocall8BE
C
	full_text6
4
2%17 = tail call i64 @_Z14get_local_sizej(i32 0) #4
2lshr8B(
&
	full_text

%18 = lshr i64 %17, 1
%i648B

	full_text
	
i64 %17
8trunc8B-
+
	full_text

%19 = trunc i64 %18 to i32
%i648B

	full_text
	
i64 %18
5icmp8B+
)
	full_text

%20 = icmp eq i32 %19, 0
%i328B

	full_text
	
i32 %19
'br8B

	full_text

br label %22
$ret8B

	full_text


ret void
Cphi8B:
8
	full_text+
)
'%23 = phi i32 [ %8, %10 ], [ %69, %65 ]
$i328B

	full_text


i32 %8
%i328B

	full_text
	
i32 %69
Cphi8B:
8
	full_text+
)
'%24 = phi i64 [ %7, %10 ], [ %68, %65 ]
$i648B

	full_text


i64 %7
%i648B

	full_text
	
i64 %68
1mul8B(
&
	full_text

%25 = mul i32 %23, %2
%i328B

	full_text
	
i32 %23
6zext8B,
*
	full_text

%26 = zext i32 %25 to i64
%i328B

	full_text
	
i32 %25
\getelementptr8BI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %0, i64 %26
%i648B

	full_text
	
i64 %26
:br8B2
0
	full_text#
!
br i1 %13, label %28, label %29
#i18B

	full_text


i1 %13
'br8B

	full_text

br label %32
Ophi8BF
D
	full_text7
5
3%30 = phi float [ 0.000000e+00, %22 ], [ %40, %32 ]
)float8B

	full_text

	float %40
Lstore8BA
?
	full_text2
0
.store float %30, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %30
+float*8B

	full_text


float* %14
:br8B2
0
	full_text#
!
br i1 %20, label %44, label %31
#i18B

	full_text


i1 %20
'br8B

	full_text

br label %47
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %41, %32 ], [ %11, %28 ]
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %11
Ophi8BF
D
	full_text7
5
3%34 = phi float [ %40, %32 ], [ 0.000000e+00, %28 ]
)float8B

	full_text

	float %40
9and8B0
.
	full_text!

%35 = and i64 %33, 4294967295
%i648B

	full_text
	
i64 %33
]getelementptr8BJ
H
	full_text;
9
7%36 = getelementptr inbounds float, float* %27, i64 %35
+float*8B

	full_text


float* %27
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !8
+float*8B

	full_text


float* %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %35
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !8
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %37, float %39, float %34)
)float8B

	full_text

	float %37
)float8B

	full_text

	float %39
)float8B

	full_text

	float %34
2add8B)
'
	full_text

%41 = add i64 %17, %35
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %35
8trunc8B-
+
	full_text

%42 = trunc i64 %41 to i32
%i648B

	full_text
	
i64 %41
7icmp8B-
+
	full_text

%43 = icmp ult i32 %42, %2
%i328B

	full_text
	
i32 %42
:br8B2
0
	full_text#
!
br i1 %43, label %32, label %29
#i18B

	full_text


i1 %43
:br8B2
0
	full_text#
!
br i1 %15, label %60, label %45
#i18B

	full_text


i1 %15
9and8	B0
.
	full_text!

%46 = and i64 %24, 4294967295
%i648	B

	full_text
	
i64 %24
'br8	B

	full_text

br label %65
Dphi8
B;
9
	full_text,
*
(%48 = phi i32 [ %58, %57 ], [ %19, %31 ]
%i328
B

	full_text
	
i32 %58
%i328
B

	full_text
	
i32 %19
Bcall8
B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6zext8
B,
*
	full_text

%49 = zext i32 %48 to i64
%i328
B

	full_text
	
i32 %48
8icmp8
B.
,
	full_text

%50 = icmp ult i64 %11, %49
%i648
B

	full_text
	
i64 %11
%i648
B

	full_text
	
i64 %49
:br8
B2
0
	full_text#
!
br i1 %50, label %51, label %57
#i18
B

	full_text


i1 %50
2add8B)
'
	full_text

%52 = add i64 %11, %49
%i648B

	full_text
	
i64 %11
%i648B

	full_text
	
i64 %49
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %5, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
Lload8BB
@
	full_text3
1
/%55 = load float, float* %14, align 4, !tbaa !8
+float*8B

	full_text


float* %14
6fadd8B,
*
	full_text

%56 = fadd float %54, %55
)float8B

	full_text

	float %54
)float8B

	full_text

	float %55
Lstore8BA
?
	full_text2
0
.store float %56, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %14
'br8B

	full_text

br label %57
2lshr8B(
&
	full_text

%58 = lshr i32 %48, 1
%i328B

	full_text
	
i32 %48
5icmp8B+
)
	full_text

%59 = icmp eq i32 %58, 0
%i328B

	full_text
	
i32 %58
:br8B2
0
	full_text#
!
br i1 %59, label %44, label %47
#i18B

	full_text


i1 %59
Hload8B>
<
	full_text/
-
+%61 = load i32, i32* %16, align 4, !tbaa !8
'i32*8B

	full_text


i32* %16
9and8B0
.
	full_text!

%62 = and i64 %24, 4294967295
%i648B

	full_text
	
i64 %24
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %4, i64 %62
%i648B

	full_text
	
i64 %62
@bitcast8B3
1
	full_text$
"
 %64 = bitcast float* %63 to i32*
+float*8B

	full_text


float* %63
Hstore8B=
;
	full_text.
,
*store i32 %61, i32* %64, align 4, !tbaa !8
%i328B

	full_text
	
i32 %61
'i32*8B

	full_text


i32* %64
'br8B

	full_text

br label %65
Dphi8B;
9
	full_text,
*
(%66 = phi i64 [ %46, %45 ], [ %62, %60 ]
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %62
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Ocall8BE
C
	full_text6
4
2%67 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
2add8B)
'
	full_text

%68 = add i64 %67, %66
%i648B

	full_text
	
i64 %67
%i648B

	full_text
	
i64 %66
8trunc8B-
+
	full_text

%69 = trunc i64 %68 to i32
%i648B

	full_text
	
i64 %68
7icmp8B-
+
	full_text

%70 = icmp ult i32 %69, %3
%i328B

	full_text
	
i32 %69
:br8B2
0
	full_text#
!
br i1 %70, label %22, label %21
#i18B

	full_text


i1 %70
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %5
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %4
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %2
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
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
,i648B!

	full_text

i64 4294967295
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 0       	
 		                      !" !! #$ ## %& %% '( '+ ** ,- ,. ,, /0 /3 24 22 56 55 78 77 9: 9; 99 <= << >? >> @A @@ BC BD BE BB FG FH FF IJ II KL KK MN MP OR QQ SU TV TT WW XY XX Z[ Z\ ZZ ]^ ]` _a __ bc bb de dd fg ff hi hj hh kl km kk np oo qr qq st sv uu wx ww yz yy {| {{ }~ } }} ?? ?
? ?? ?? ?? ?? ?
? ?? ?? ?? ?? ?? ?? ?? >? ? ? b	? 
? ?? y? %	? 	? !	? K    
	       ?  ?   "! $# & (B +* - . 0F 3 4B 62 8% :7 ;9 =7 ?> A< C@ D5 E G7 HF JI LK N P Ro U VT Y [X \Z ^ `X a_ cb e gd if jh l mT po rq t v xw zy |u ~{ Q ?w ?? ?? ?? ?? ?? ?   ' )' *) 2/ O/ 1M 2M *O uO Q1 T? ?S ?] _] o? ? n os Os T ??  ?? ?? ?? ?? ??? ?? ? ?? ? ?? ?B ?? B ??  ?? W ?? W	? 	? ? W	? o? ?	? 7	? Q	? w? *	? 5? ? ? 	? 	? q? ?"
MatVecMulCoalesced2"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f32"
_Z14get_local_sizej"
_Z7barrierj"
_Z14get_num_groupsj*?
+nvidia-4.2-MatVecMul-MatVecMulCoalesced2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?
 
transfer_bytes_log1p
?9?A

devmap_label


transfer_bytes	
????

wgsize_log1p
?9?A