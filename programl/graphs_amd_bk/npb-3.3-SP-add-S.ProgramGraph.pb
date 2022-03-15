
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
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
4icmpB,
*
	full_text

%14 = icmp sgt i32 %8, %4
"i32B

	full_text


i32 %8
6truncB-
+
	full_text

%15 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
5icmpB-
+
	full_text

%16 = icmp sgt i32 %15, %3
#i32B

	full_text
	
i32 %15
-orB'
%
	full_text

%17 = or i1 %14, %16
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %16
5icmpB-
+
	full_text

%18 = icmp sgt i32 %13, %2
#i32B

	full_text
	
i32 %13
-orB'
%
	full_text

%19 = or i1 %17, %18
!i1B

	full_text


i1 %17
!i1B

	full_text


i1 %18
8brB2
0
	full_text#
!
br i1 %19, label %54, label %20
!i1B

	full_text


i1 %19
Wbitcast8BJ
H
	full_text;
9
7%21 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%22 = bitcast double* %1 to [13 x [13 x [5 x double]]]*
0shl8B'
%
	full_text

%23 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
1shl8B(
&
	full_text

%25 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
1shl8B(
&
	full_text

%27 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
¢getelementptr8Bé
ã
	full_text~
|
z%29 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%30 = load double, double* %29, align 8, !tbaa !8
-double*8B

	full_text

double* %29
¢getelementptr8Bé
ã
	full_text~
|
z%31 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%32 = load double, double* %31, align 8, !tbaa !8
-double*8B

	full_text

double* %31
7fadd8B-
+
	full_text

%33 = fadd double %30, %32
+double8B

	full_text


double %30
+double8B

	full_text


double %32
Nstore8BC
A
	full_text4
2
0store double %33, double* %29, align 8, !tbaa !8
+double8B

	full_text


double %33
-double*8B

	full_text

double* %29
¢getelementptr8Bé
ã
	full_text~
|
z%34 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
¢getelementptr8Bé
ã
	full_text~
|
z%36 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%37 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
7fadd8B-
+
	full_text

%38 = fadd double %35, %37
+double8B

	full_text


double %35
+double8B

	full_text


double %37
Nstore8BC
A
	full_text4
2
0store double %38, double* %34, align 8, !tbaa !8
+double8B

	full_text


double %38
-double*8B

	full_text

double* %34
¢getelementptr8Bé
ã
	full_text~
|
z%39 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
¢getelementptr8Bé
ã
	full_text~
|
z%41 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%42 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
7fadd8B-
+
	full_text

%43 = fadd double %40, %42
+double8B

	full_text


double %40
+double8B

	full_text


double %42
Nstore8BC
A
	full_text4
2
0store double %43, double* %39, align 8, !tbaa !8
+double8B

	full_text


double %43
-double*8B

	full_text

double* %39
¢getelementptr8Bé
ã
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
¢getelementptr8Bé
ã
	full_text~
|
z%46 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%47 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
7fadd8B-
+
	full_text

%48 = fadd double %45, %47
+double8B

	full_text


double %45
+double8B

	full_text


double %47
Nstore8BC
A
	full_text4
2
0store double %48, double* %44, align 8, !tbaa !8
+double8B

	full_text


double %48
-double*8B

	full_text

double* %44
¢getelementptr8Bé
ã
	full_text~
|
z%49 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %24, i64 %26, i64 %28, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%50 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
¢getelementptr8Bé
ã
	full_text~
|
z%51 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %22, i64 %24, i64 %26, i64 %28, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
7fadd8B-
+
	full_text

%53 = fadd double %50, %52
+double8B

	full_text


double %50
+double8B

	full_text


double %52
Nstore8BC
A
	full_text4
2
0store double %53, double* %49, align 8, !tbaa !8
+double8B

	full_text


double %53
-double*8B

	full_text

double* %49
'br8B

	full_text

br label %54
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 0        		 
 

                     !    "# "" $% $$ &' && () (( *+ ** ,- ,. ,/ ,0 ,, 12 11 34 35 36 37 33 89 88 :; :< :: => =? == @A @B @C @D @@ EF EE GH GI GJ GK GG LM LL NO NP NN QR QS QQ TU TV TW TX TT YZ YY [\ [] [^ [_ [[ `a `` bc bd bb ef eg ee hi hj hk hl hh mn mm op oq or os oo tu tt vw vx vv yz y{ yy |} |~ | |	Ä || ÅÇ ÅÅ ÉÑ É
Ö É
Ü É
á ÉÉ àâ àà äã ä
å ää çé ç
è çç ê	í 	ì î 	ï ñ    	 
           !  # %$ '
 )( + -" .& /* 0, 2 4" 5& 6* 73 91 ;8 <: >, ? A" B& C* D@ F H" I& J* KG ME OL PN R@ S U" V& W* XT Z \" ]& ^* _[ aY c` db fT g i" j& k* lh n p" q& r* so um wt xv zh { }" ~& * Ä| Ç Ñ" Ö& Ü* áÉ âÅ ãà åä é| è ë ê ë ë óó óó 	 óó 	 óó ò 	ô 	ö T	ö [õ 	ú 	ú 	ú 
	ú @	ú G	ù h	ù o	û |
û É	ü  	ü "	ü $	ü &	ü (	ü *	† ,	† 3"
add"
_Z13get_global_idj*Ü
npb-SP-add.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
ãèYA

transfer_bytes
∏ä1

devmap_label
 

wgsize_log1p
ãèYA

wgsize
<