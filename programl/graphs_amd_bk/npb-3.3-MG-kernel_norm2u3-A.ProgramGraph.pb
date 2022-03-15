

[external]
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 0) #4
KcallBC
A
	full_text4
2
0%10 = tail call i64 @_Z12get_local_idj(i32 0) #4
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
2addB+
)
	full_text

%13 = add nsw i32 %1, -1
6icmpB.
,
	full_text

%14 = icmp sgt i32 %13, %12
#i32B

	full_text
	
i32 %13
#i32B

	full_text
	
i32 %12
8brB2
0
	full_text#
!
br i1 %14, label %17, label %15
!i1B

	full_text


i1 %14
Ocall8BE
C
	full_text6
4
2%16 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %45
/add8B&
$
	full_text

%18 = add i64 %9, 1
$i648B

	full_text


i64 %9
8trunc8B-
+
	full_text

%19 = trunc i64 %18 to i32
%i648B

	full_text
	
i64 %18
Ncall8BD
B
	full_text5
3
1%20 = tail call i64 @_Z13get_global_idj(i32 1) #4
0add8B'
%
	full_text

%21 = add i64 %20, 1
%i648B

	full_text
	
i64 %20
8trunc8B-
+
	full_text

%22 = trunc i64 %21 to i32
%i648B

	full_text
	
i64 %21
5mul8B,
*
	full_text

%23 = mul nsw i32 %22, %2
%i328B

	full_text
	
i32 %22
2add8B)
'
	full_text

%24 = add i32 %23, %19
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %19
1mul8B(
&
	full_text

%25 = mul i32 %24, %1
%i328B

	full_text
	
i32 %24
Ocall8BE
C
	full_text6
4
2%26 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %27
Dphi8B;
9
	full_text,
*
(%28 = phi i32 [ %12, %17 ], [ %43, %27 ]
%i328B

	full_text
	
i32 %12
%i328B

	full_text
	
i32 %43
Dphi8B;
9
	full_text,
*
(%29 = phi i64 [ %11, %17 ], [ %42, %27 ]
%i648B

	full_text
	
i64 %11
%i648B

	full_text
	
i64 %42
Pphi8BG
E
	full_text8
6
4%30 = phi double [ 0.000000e+00, %17 ], [ %36, %27 ]
+double8B

	full_text


double %36
Pphi8BG
E
	full_text8
6
4%31 = phi double [ 0.000000e+00, %17 ], [ %39, %27 ]
+double8B

	full_text


double %39
6add8B-
+
	full_text

%32 = add nsw i32 %28, %25
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %25
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
^getelementptr8BK
I
	full_text<
:
8%34 = getelementptr inbounds double, double* %0, i64 %33
%i648B

	full_text
	
i64 %33
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
icall8B_
]
	full_textP
N
L%36 = tail call double @llvm.fmuladd.f64(double %35, double %35, double %30)
+double8B

	full_text


double %35
+double8B

	full_text


double %35
+double8B

	full_text


double %30
Lcall8BB
@
	full_text3
1
/%37 = tail call double @_Z4fabsd(double %35) #4
+double8B

	full_text


double %35
;fcmp8B1
/
	full_text"
 
%38 = fcmp ogt double %37, %31
+double8B

	full_text


double %37
+double8B

	full_text


double %31
Jselect8B>
<
	full_text/
-
+%39 = select i1 %38, double %37, double %31
#i18B

	full_text


i1 %38
+double8B

	full_text


double %37
+double8B

	full_text


double %31
1shl8B(
&
	full_text

%40 = shl i64 %29, 32
%i648B

	full_text
	
i64 %29
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
2add8B)
'
	full_text

%42 = add i64 %26, %41
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %41
8trunc8B-
+
	full_text

%43 = trunc i64 %42 to i32
%i648B

	full_text
	
i64 %42
8icmp8B.
,
	full_text

%44 = icmp sgt i32 %13, %43
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %43
:br8B2
0
	full_text#
!
br i1 %44, label %27, label %45
#i18B

	full_text


i1 %44
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %16, %15 ], [ %26, %27 ]
%i648B

	full_text
	
i64 %16
%i648B

	full_text
	
i64 %26
Pphi8BG
E
	full_text8
6
4%47 = phi double [ 0.000000e+00, %15 ], [ %39, %27 ]
+double8B

	full_text


double %39
Pphi8BG
E
	full_text8
6
4%48 = phi double [ 0.000000e+00, %15 ], [ %36, %27 ]
+double8B

	full_text


double %36
8trunc8B-
+
	full_text

%49 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
1shl8B(
&
	full_text

%50 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%51 = ashr exact i64 %50, 32
%i648B

	full_text
	
i64 %50
^getelementptr8BK
I
	full_text<
:
8%52 = getelementptr inbounds double, double* %6, i64 %51
%i648B

	full_text
	
i64 %51
Nstore8BC
A
	full_text4
2
0store double %48, double* %52, align 8, !tbaa !8
+double8B

	full_text


double %48
-double*8B

	full_text

double* %52
^getelementptr8BK
I
	full_text<
:
8%53 = getelementptr inbounds double, double* %7, i64 %51
%i648B

	full_text
	
i64 %51
Nstore8BC
A
	full_text4
2
0store double %47, double* %53, align 8, !tbaa !8
+double8B

	full_text


double %47
-double*8B

	full_text

double* %53
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
2lshr8B(
&
	full_text

%54 = lshr i64 %46, 1
%i648B

	full_text
	
i64 %46
8trunc8B-
+
	full_text

%55 = trunc i64 %54 to i32
%i648B

	full_text
	
i64 %54
6icmp8B,
*
	full_text

%56 = icmp sgt i32 %55, 0
%i328B

	full_text
	
i32 %55
:br8B2
0
	full_text#
!
br i1 %56, label %57, label %59
#i18B

	full_text


i1 %56
Abitcast8B4
2
	full_text%
#
!%58 = bitcast double* %53 to i64*
-double*8B

	full_text

double* %53
'br8B

	full_text

br label %61
5icmp8B+
)
	full_text

%60 = icmp eq i32 %49, 0
%i328B

	full_text
	
i32 %49
:br8B2
0
	full_text#
!
br i1 %60, label %83, label %98
#i18B

	full_text


i1 %60
Dphi8B;
9
	full_text,
*
(%62 = phi i32 [ %55, %57 ], [ %81, %80 ]
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %81
8icmp8B.
,
	full_text

%63 = icmp sgt i32 %62, %49
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %49
:br8B2
0
	full_text#
!
br i1 %63, label %64, label %80
#i18B

	full_text


i1 %63
6add8B-
+
	full_text

%65 = add nsw i32 %62, %49
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %49
6sext8B,
*
	full_text

%66 = sext i32 %65 to i64
%i328B

	full_text
	
i32 %65
^getelementptr8BK
I
	full_text<
:
8%67 = getelementptr inbounds double, double* %6, i64 %66
%i648B

	full_text
	
i64 %66
Nload8BD
B
	full_text5
3
1%68 = load double, double* %67, align 8, !tbaa !8
-double*8B

	full_text

double* %67
Nload8BD
B
	full_text5
3
1%69 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
7fadd8B-
+
	full_text

%70 = fadd double %68, %69
+double8B

	full_text


double %68
+double8B

	full_text


double %69
Nstore8BC
A
	full_text4
2
0store double %70, double* %52, align 8, !tbaa !8
+double8B

	full_text


double %70
-double*8B

	full_text

double* %52
Nload8BD
B
	full_text5
3
1%71 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
^getelementptr8BK
I
	full_text<
:
8%72 = getelementptr inbounds double, double* %7, i64 %66
%i648B

	full_text
	
i64 %66
Nload8BD
B
	full_text5
3
1%73 = load double, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
;fcmp8B1
/
	full_text"
 
%74 = fcmp ogt double %71, %73
+double8B

	full_text


double %71
+double8B

	full_text


double %73
Dselect8B8
6
	full_text)
'
%%75 = select i1 %74, i32 %49, i32 %65
#i18B

	full_text


i1 %74
%i328B

	full_text
	
i32 %49
%i328B

	full_text
	
i32 %65
6sext8B,
*
	full_text

%76 = sext i32 %75 to i64
%i328B

	full_text
	
i32 %75
^getelementptr8BK
I
	full_text<
:
8%77 = getelementptr inbounds double, double* %7, i64 %76
%i648B

	full_text
	
i64 %76
Abitcast8B4
2
	full_text%
#
!%78 = bitcast double* %77 to i64*
-double*8B

	full_text

double* %77
Hload8B>
<
	full_text/
-
+%79 = load i64, i64* %78, align 8, !tbaa !8
'i64*8B

	full_text


i64* %78
Hstore8B=
;
	full_text.
,
*store i64 %79, i64* %58, align 8, !tbaa !8
%i648B

	full_text
	
i64 %79
'i64*8B

	full_text


i64* %58
'br8B

	full_text

br label %80
Bcall8	B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
2lshr8	B(
&
	full_text

%81 = lshr i32 %62, 1
%i328	B

	full_text
	
i32 %62
5icmp8	B+
)
	full_text

%82 = icmp eq i32 %81, 0
%i328	B

	full_text
	
i32 %81
:br8	B2
0
	full_text#
!
br i1 %82, label %59, label %61
#i18	B

	full_text


i1 %82
Mcall8
BC
A
	full_text4
2
0%84 = tail call i64 @_Z12get_group_idj(i32 1) #4
Ocall8
BE
C
	full_text6
4
2%85 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
2mul8
B)
'
	full_text

%86 = mul i64 %85, %84
%i648
B

	full_text
	
i64 %85
%i648
B

	full_text
	
i64 %84
1add8
B(
&
	full_text

%87 = add i64 %86, %9
%i648
B

	full_text
	
i64 %86
$i648
B

	full_text


i64 %9
@bitcast8
B3
1
	full_text$
"
 %88 = bitcast double* %6 to i64*
Hload8
B>
<
	full_text/
-
+%89 = load i64, i64* %88, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %88
1shl8
B(
&
	full_text

%90 = shl i64 %87, 32
%i648
B

	full_text
	
i64 %87
9ashr8
B/
-
	full_text 

%91 = ashr exact i64 %90, 32
%i648
B

	full_text
	
i64 %90
^getelementptr8
BK
I
	full_text<
:
8%92 = getelementptr inbounds double, double* %4, i64 %91
%i648
B

	full_text
	
i64 %91
Abitcast8
B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8
B

	full_text

double* %92
Hstore8
B=
;
	full_text.
,
*store i64 %89, i64* %93, align 8, !tbaa !8
%i648
B

	full_text
	
i64 %89
'i64*8
B

	full_text


i64* %93
@bitcast8
B3
1
	full_text$
"
 %94 = bitcast double* %7 to i64*
Hload8
B>
<
	full_text/
-
+%95 = load i64, i64* %94, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %94
^getelementptr8
BK
I
	full_text<
:
8%96 = getelementptr inbounds double, double* %5, i64 %91
%i648
B

	full_text
	
i64 %91
Abitcast8
B4
2
	full_text%
#
!%97 = bitcast double* %96 to i64*
-double*8
B

	full_text

double* %96
Hstore8
B=
;
	full_text.
,
*store i64 %95, i64* %97, align 8, !tbaa !8
%i648
B

	full_text
	
i64 %95
'i64*8
B

	full_text


i64* %97
'br8
B

	full_text

br label %98
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %2
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %7
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1
4double8B&
$
	full_text

double 0.000000e+00       	 
                      " !# !! $% $& $$ '( '' )* )) +, +- ++ ./ .. 01 00 23 22 45 46 47 44 89 88 :; :< :: => =? =@ == AB AA CD CC EF EG EE HI HH JK JL JJ MN MP OQ OO RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd ce cc ff gh gg ij ii kl kk mn mp oo qs rr tu tw vx vv yz y{ yy |} | ~	Ä ~~ ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ ÖÖ áà áá âä â
ã ââ åç å
é åå èê èè ë
í ëë ìî ìì ïñ ï
ó ïï òô ò
ö ò
õ òò úù úú û
ü ûû †° †† ¢£ ¢¢ §• §
¶ §§ ß® ©™ ©© ´¨ ´´ ≠Æ ≠Ø ∞∞ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∑ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈≈ ∆« ∆∆ »
… »»  À    ÃÕ Ã
Œ ÃÃ œ— 	— 	“ ” æ‘ »’ 0÷ \÷ É÷ ∑◊ a◊ ë◊ û◊ ≈   	 
          "H # %E &4 (= *! , -+ /. 10 32 52 6' 72 98 ;) <: >8 ?) @$ BA D FC GE I KH LJ N P Q= S4 U W YX [Z ]T _\ `Z bR da eO hg ji lk na pV sr ui w© xv zV {y }v V Ä~ ÇÅ ÑÉ Ü\ àÖ äá ãâ ç\ éa êÅ íë îè ñì óï ôV ö~ õò ùú üû °† £¢ •o ¶v ™© ¨´ Æ∞ ≤Ø ≥± µ ∂∑ π¥ ª∫ Ωº øæ ¡∏ √¿ ƒ≈ «º …» À∆ Õ  Œ    ! OM !M Om om rq vt Øt –| ~| ®œ –ß ®≠ r≠ v ﬂﬂ ÿÿ ŸŸ – €€ ⁄⁄ ‹‹ ﬁﬁ ›› ŸŸ  ››  ÿÿ f ﬁﬁ f ⁄⁄  ›› 8 ‹‹ 8Ø ŸŸ Ø∞ ﬂﬂ ∞® ﬁﬁ ®4 €€ 4	‡ A	‡ C	‡ X	‡ Z
‡ ∫
‡ º· · f· ®
· ©· Ø	‚ 	‚ 	‚ 	‚ g„ „ „ „ 	„ k	„ r
„ ´„ ∞	‰ Â 'Â )Â RÂ T"
kernel_norm2u3"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f64"

_Z4fabsd"
_Z14get_local_sizej"
_Z7barrierj"
_Z14get_num_groupsj*ë
npb-MG-kernel_norm2u3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ç
 
transfer_bytes_log1p
îﬁüA

devmap_label


transfer_bytes	
∞ÊÃ„

wgsize
Ä

wgsize_log1p
îﬁüA